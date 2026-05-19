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

from data_utils.data_loader import Comma_CAN_Temporal, Comma_Instance

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
        default="Predict a driving action from the current scene.",
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


def tensor_to_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    target_size = (args.width, args.height)
    policy = SmolVLAPolicy.from_pretrained(args.model_id).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    input_features = config_get(policy.config, "input_features", {})
    visual_keys, state_key = split_feature_keys(input_features)
    state_dim = 0
    if state_key is not None:
        state_dim = get_feature_shape(input_features[state_key])[0]

    if not visual_keys:
        raise RuntimeError("Could not find any visual input features in the SmolVLA config.")

    dataset_cls = Comma_CAN_Temporal if args.use_temporal_state else Comma_Instance
    dataset = dataset_cls(
        args.chunk_path,
        target_size=target_size,
        future_time=args.future_time,
        policy_visual_keys=visual_keys,
        policy_state_key=state_key,
        policy_state_dim=state_dim if state_key is not None else None,
        task=args.task,
        robot_type=args.robot_type,
    )

    print(f"Loaded policy: {args.model_id}")
    print(f"Device: {device}")
    print(f"Expected visual keys: {visual_keys}")
    if state_key is not None:
        print(f"Expected state key: {state_key} with dim={state_dim}")
    print(f"Using dataset: {dataset_cls.__name__}")
    print()

    results: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(dataset):
        if sample_index >= args.num_samples:
            break

        frame = sample["policy_observation"]
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
