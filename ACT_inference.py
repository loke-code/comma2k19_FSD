import argparse
import torch
from pathlib import Path
from typing import Any

# Import necessary components from the project
from data_utils.data_loader import Comma_CAN_Temporal, Comma_Instance
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from huggingface_hub import snapshot_download

# Default values for demonstration
DEFAULT_HISTORY_OFFSETS = [0.1, 0.4, 1.0, 2.0]

def resolve_device(device_arg: str) -> torch.device:
    """Resolve the appropriate torch device."""
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def config_get(config: Any, key: str, default: Any = None) -> Any:
    """Helper to get config values from dict or object."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

def get_feature_shape(feature: Any) -> tuple[int, ...]:
    """Get the shape of a feature."""
    shape = config_get(feature, "shape", ())
    return tuple(shape) if shape is not None else ()

def split_feature_keys(input_features: dict[str, Any]) -> tuple[list[str], str | None]:
    """Split input features into visual and state keys."""
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

def ensure_model_available(model_id: str) -> str:
    """Ensure model is available locally, downloading only if necessary."""
    model_path = Path(model_id)
    if model_path.exists():
        print(f"Using local checkpoint: {model_path}")
        return str(model_path)
    
    # Check for the common 'last' checkpoint structure if only the base dir is provided
    last_checkpoint = model_path / "checkpoints" / "last" / "pretrained_model"
    if last_checkpoint.exists():
        print(f"Using local 'last' checkpoint: {last_checkpoint}")
        return str(last_checkpoint)

    print(f"Local checkpoint not found at {model_id}. Checking Hugging Face cache...")
    try:
        local_model_path = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_files_only=True,
        )
        return local_model_path
    except Exception:
        print(f"Model not found in local cache. Downloading: {model_id}")
        local_model_path = snapshot_download(
            repo_id=model_id,
            repo_type="model",
        )
        return local_model_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained ACT policy.")
    parser.add_argument(
        "--chunk-path",
        type=Path,
        required=True,
        help="Path to a comma2k19 extracted chunk, e.g. comma2k19_data/extracted/Chunk_1",
    )
    parser.add_argument(
        "--model-id",
        default="outputs/train/act_comma2k19/checkpoints/last/pretrained_model",
        help="Path to trained ACT checkpoint (e.g. outputs/train/act_comma2k19/checkpoints/last/pretrained_model)",
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
        help="How many dataset samples (frames) to run through the policy",
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
        "--use-temporal-state",
        action="store_true",
        help="Use Comma_CAN_Temporal and build a richer state vector from CAN history",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    target_size = (args.width, args.height)

    # 1. Load the trained ACT policy
    model_source = ensure_model_available(args.model_id)
    policy = ACTPolicy.from_pretrained(model_source).to(device).eval()

    # Get pre- and post-processors based on policy config
    preprocess, postprocess = make_act_pre_post_processors(
        policy.config)

    # Determine visual and state keys from policy config
    input_features = config_get(policy.config, "input_features", {})
    visual_keys, state_key = split_feature_keys(input_features)
    state_dim = 0
    if state_key is not None:
        state_dim = get_feature_shape(input_features[state_key])[0]

    if not visual_keys:
        raise RuntimeError("Could not find any visual input features in the policy config.")

    # Determine history offsets for the data loader
    history_offsets: list[float]
    if args.use_temporal_state:
        history_offsets = DEFAULT_HISTORY_OFFSETS
    else:
        history_offsets = []

    # 2. Set up the data loader for individual frames
    dataset_cls = Comma_CAN_Temporal if args.use_temporal_state else Comma_Instance
    dataset = dataset_cls(
        args.chunk_path,
        target_size=target_size,
        future_time=args.future_time,
        history_offsets=history_offsets,
        policy_visual_keys=visual_keys,
        policy_state_key=state_key,
        policy_state_dim=state_dim if state_key is not None else None,
        robot_type="comma2k19_car",
    )

    print(f"Running inference with model: {model_source}")
    print(f"Device: {device}")
    print(f"Using dataset: {dataset_cls.__name__}")
    print(f"Expected visual keys: {visual_keys}")
    if state_key is not None:
        print(f"Expected state key: {state_key} with dim={state_dim}")
    print("-" * 30)

    # 3. Perform inference on a few sample frames
    with torch.inference_mode(): # Disable gradient calculations for inference
        for sample_index, sample in enumerate(dataset):
            if sample_index >= args.num_samples:
                break

            # Get the observation from the data loader
            frame_observation = sample["policy_observation"]

            # Preprocess the observation for the policy
            processed_observation = preprocess(frame_observation)

            # Get action prediction from the policy
            action = policy.select_action(processed_observation)

            # Postprocess the action (e.g., denormalize)
            action = postprocess(action)

            # Print results
            print(f"Sample {sample_index + 1}:")
            print(f"  Input State (partial): {frame_observation.get(state_key, 'N/A')}")
            print(f"  Predicted Action: {action.detach().cpu().tolist()}")
            print("-" * 30)

if __name__ == "__main__":
    main()
