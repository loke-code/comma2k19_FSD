#!/usr/bin/env python3
"""
Run offline SmolVLA inference on comma2k19 samples using the local data loader.

This script consumes policy-ready observations emitted by
`data_utils/data_loader.py`.

Notes:
- The public SmolVLA examples load checkpoints with
  `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")` via LeRobot.
- The data loader handles RGB conversion, state construction, and SmolVLA /
  LeRobot-style observation mapping.
- The base SmolVLA checkpoint is intended to be fine-tuned on your own task, so
  predictions on raw comma2k19 driving data will likely not be meaningful until
  you fine-tune a policy on a compatible dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from data_utils.data_loader import Comma_CAN_Temporal, Comma_Instance

DEFAULT_TASK_PROMPT = (
    "Given the current driving scene and vehicle state, predict the next driving action."
)
DEFAULT_HISTORY_OFFSETS = [0.1, 0.4, 1.0, 2.0]

try:
    from lerobot.policies import make_pre_post_processors
except ImportError:
    from lerobot.policies.factory import make_pre_post_processors

try:
    from lerobot.policies.smolvla import SmolVLAPolicy
except ImportError:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-path",
        type=Path,
        required=True,
        help="Path to a comma2k19 extracted chunk, e.g. comma2k19_data/extracted/Chunk_1",
    )
    parser.add_argument(
        "--model-id",
        default="lerobot/smolvla_base",
        help="Hugging Face model id or local LeRobot policy checkpoint",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device to run on: "auto", "cpu", "cuda", or "mps"',
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="How many dataset samples to run through the policy",
    )
    parser.add_argument(
        "--future-time",
        type=float,
        default=1.0,
        help="Future prediction horizon forwarded to the local data loader",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Resized image height for the local data loader",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Resized image width for the local data loader",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK_PROMPT,
        help="Language instruction passed to the policy preprocessor",
    )
    parser.add_argument(
        "--robot-type",
        default="comma2k19_car",
        help="Optional robot/embodiment tag included in the observation frame",
    )
    parser.add_argument(
        "--use-temporal-state",
        action="store_true",
        help="Use Comma_CAN_Temporal and build a richer state vector from CAN history",
    )
    parser.add_argument(
        "--history-offsets",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional past-time offsets in seconds for CAN history, for example "
            "--history-offsets 0.1 0.4 1.0 2.0. If omitted, temporal mode uses "
            f"{DEFAULT_HISTORY_OFFSETS}."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional path to save per-sample predictions as JSONL",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def get_feature_shape(feature: Any) -> tuple[int, ...]:
    shape = config_get(feature, "shape", ())
    return tuple(shape) if shape is not None else ()


def split_feature_keys(input_features: dict[str, Any]) -> tuple[list[str], str | None]:
    visual_keys: list[str] = []
    state_key: str | None = None

    for key, feature in input_features.items():
        feature_type = str(config_get(feature, "type", ""))
        if "VISUAL" in feature_type:
            visual_keys.append(key)
        elif "STATE" in feature_type and state_key is None:
            state_key = key

    visual_keys.sort()
    return visual_keys, state_key


def validate_visual_feature_shapes(
    input_features: dict[str, Any],
    visual_keys: list[str],
    target_size: tuple[int, int],
) -> tuple[int, int, int]:
    visual_shapes = [get_feature_shape(input_features[key]) for key in visual_keys]
    if not visual_shapes:
        raise RuntimeError("No visual feature shapes found in the policy config.")

    first_shape = visual_shapes[0]
    if len(first_shape) != 3:
        raise RuntimeError(f"Expected visual features with shape (C, H, W), got {first_shape}.")

    for key, shape in zip(visual_keys, visual_shapes):
        if shape != first_shape:
            raise RuntimeError(
                f"Visual feature shapes must match across cameras. "
                f"Got {key}={shape} and first shape={first_shape}."
            )

    channels, expected_height, expected_width = first_shape
    requested_width, requested_height = target_size
    if channels != 3:
        raise RuntimeError(f"Expected 3-channel visual inputs, got {channels}.")
    if (requested_height, requested_width) != (expected_height, expected_width):
        raise ValueError(
            "Requested loader image size does not match the checkpoint. "
            f"Loader target_size=(width={requested_width}, height={requested_height}), "
            f"but policy expects (C={channels}, H={expected_height}, W={expected_width})."
        )
    return channels, expected_height, expected_width


def validate_policy_observation(
    observation: dict[str, Any],
    visual_keys: list[str],
    state_key: str | None,
    state_dim: int,
    expected_visual_shape: tuple[int, int, int],
) -> None:
    for key in visual_keys:
        if key not in observation:
            raise KeyError(f"Missing required visual key '{key}' in policy observation.")
        value = observation[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Visual key '{key}' must be a torch.Tensor, got {type(value).__name__}.")
        if tuple(value.shape) != expected_visual_shape:
            raise ValueError(
                f"Visual key '{key}' has shape {tuple(value.shape)}, "
                f"expected {expected_visual_shape}."
            )

    if state_key is not None:
        if state_key not in observation:
            raise KeyError(f"Missing required state key '{state_key}' in policy observation.")
        state_value = observation[state_key]
        if not isinstance(state_value, torch.Tensor):
            raise TypeError(
                f"State key '{state_key}' must be a torch.Tensor, got {type(state_value).__name__}."
            )
        if tuple(state_value.shape) != (state_dim,):
            raise ValueError(
                f"State key '{state_key}' has shape {tuple(state_value.shape)}, "
                f"expected {(state_dim,)}."
            )


def ensure_model_available(model_id: str) -> str:
    model_path = Path(model_id)
    if model_path.exists():
        return str(model_path)

    print(f"Checking local cache for model: {model_id}")
    try:
        local_model_path = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_files_only=True,
        )
        print(f"Using cached model at: {local_model_path}")
        return local_model_path
    except Exception:
        pass

    print(f"Model not found in local cache. Downloading: {model_id}")
    local_model_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
    )
    print(f"Downloaded model to: {local_model_path}")
    return local_model_path


def tensor_to_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    target_size = (args.width, args.height)
    model_source = ensure_model_available(args.model_id)
    policy = SmolVLAPolicy.from_pretrained(model_source).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_source,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    input_features = config_get(policy.config, "input_features", {})
    visual_keys, state_key = split_feature_keys(input_features)
    state_dim = 0
    if state_key is not None:
        state_dim = get_feature_shape(input_features[state_key])[0]

    if not visual_keys:
        raise RuntimeError("Could not find any visual input features in the SmolVLA config.")
    expected_visual_shape = validate_visual_feature_shapes(
        input_features=input_features,
        visual_keys=visual_keys,
        target_size=target_size,
    )

    history_offsets: list[float]
    if args.history_offsets is not None:
        history_offsets = args.history_offsets
    elif args.use_temporal_state:
        history_offsets = DEFAULT_HISTORY_OFFSETS
    else:
        history_offsets = []

    dataset_cls = Comma_CAN_Temporal if args.use_temporal_state else Comma_Instance
    dataset = dataset_cls(
        args.chunk_path,
        target_size=target_size,
        future_time=args.future_time,
        history_offsets=history_offsets,
        policy_visual_keys=visual_keys,
        policy_state_key=state_key,
        policy_state_dim=state_dim if state_key is not None else None,
        task=args.task,
        robot_type=args.robot_type,
    )

    print(f"Loaded policy: {args.model_id}")
    print(f"Model source: {model_source}")
    print(f"Device: {device}")
    print(f"Expected visual keys: {visual_keys}")
    print(f"Expected visual shape: {expected_visual_shape}")
    if state_key is not None:
        print(f"Expected state key: {state_key} with dim={state_dim}")
    print(f"Using dataset: {dataset_cls.__name__}")
    print(f"History offsets: {history_offsets}")
    print()

    results: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(dataset):
        if sample_index >= args.num_samples:
            break

        frame = sample["policy_observation"]
        validate_policy_observation(
            observation=frame,
            visual_keys=visual_keys,
            state_key=state_key,
            state_dim=state_dim,
            expected_visual_shape=expected_visual_shape,
        )
        processed = preprocess(frame)

        with torch.inference_mode():
            action = policy.select_action(processed)
            action = postprocess(action)

        result = {
            "sample_index": sample_index,
            "pred_action": tensor_to_list(action),
            "gt_future": {
                "speed": float(sample["y_speed"].item()),
                "steer": float(sample["y_steer"].item()),
            },
            "input_state": tensor_to_list(frame.get(state_key)) if state_key else None,
        }
        results.append(result)

        print(f"Sample {sample_index}")
        print(f"  input_state: {result['input_state']}")
        print(f"  pred_action: {result['pred_action']}")
        print(f"  gt_future: {result['gt_future']}")
        print()

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        print(f"Saved predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
