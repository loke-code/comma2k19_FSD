#!/usr/bin/env python3
"""
Convert comma2k19 extracted chunks into a local LeRobotDataset for ACT training.

Why this exists:
- Online streaming from video.hevc means every training epoch repeatedly decodes
  comma2k19 video and interpolates CAN data.
- ACT training works best with a materialized LeRobotDataset, where images,
  states, actions, episode metadata, and normalization stats are saved once.

Output feature schema:
- observation.images.front: RGB camera frame, shape (H, W, 3)
- observation.state: vehicle input state, shape (D,)
- action: future driving target, shape (2,)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

DEFAULT_TASK = "Predict the future speed and steering command from the current driving scene."
DEFAULT_HISTORY_OFFSETS = [0.1, 0.4, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-path",
        type=Path,
        required=True,
        help="Path to extracted comma2k19 chunk, for example comma2k19_data/extracted/Chunk_1",
    )
    parser.add_argument(
        "--repo-id",
        default="local/comma2k19_act",
        help="LeRobot dataset repo id. Local ids like local/comma2k19_act are fine.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("lerobot_datasets"),
        help="Directory where the local LeRobotDataset will be written.",
    )
    parser.add_argument("--width", type=int, default=256, help="Saved image width.")
    parser.add_argument("--height", type=int, default=256, help="Saved image height.")
    parser.add_argument("--fps", type=int, default=20, help="Dataset FPS stored in LeRobot metadata.")
    parser.add_argument(
        "--future-time",
        type=float,
        default=1.0,
        help="Target action horizon in seconds. The action is speed/steer at t + future_time.",
    )
    parser.add_argument(
        "--history-offsets",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Past CAN offsets in seconds. Use no values for no history, or omit the flag "
            f"to use the default {DEFAULT_HISTORY_OFFSETS}."
        ),
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable CAN history and store only current speed/steer in observation.state.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Task string saved with every LeRobot frame.",
    )
    parser.add_argument(
        "--use-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store visual observations as videos. Use --no-use-videos for image files.",
    )
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help="Use LeRobot streaming video encoding if supported by your installed version.",
    )
    parser.add_argument(
        "--batch-encoding-size",
        type=int,
        default=1,
        help="Number of episodes to batch before video encoding.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Async image writer threads used by LeRobotDataset.create.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Async image writer processes used by LeRobotDataset.create.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output dataset directory before conversion.",
    )
    return parser.parse_args()


def import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            raise SystemExit(
                "LeRobot is not installed in this environment. Install it first, then rerun this script.\n"
                "Example: pip install lerobot"
            ) from exc
    return LeRobotDataset


def resolve_history_offsets(args: argparse.Namespace) -> list[float]:
    if args.no_history:
        return []
    if args.history_offsets is None:
        return DEFAULT_HISTORY_OFFSETS
    return args.history_offsets


def import_comma_dataset():
    try:
        from data_utils.data_loader import CommaLeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "Could not import the local comma2k19 data loader. Run this script from the "
            "comma2k19_FSD project directory or ensure it is on PYTHONPATH."
        ) from exc
    return CommaLeRobotDataset


def tensor_image_to_uint8_hwc(image: Any) -> Any:
    import numpy as np

    if image.ndim != 3:
        raise ValueError(f"Expected image tensor shape (3, H, W), got {tuple(image.shape)}")
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def make_features(height: int, width: int, state_dim: int, use_videos: bool) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if use_videos else "image"
    state_names = ["speed", "steer"]
    for idx in range((state_dim - 2) // 2):
        state_names.extend([f"speed_history_{idx}", f"steer_history_{idx}"])

    return {
        "observation.images.front": {
            "dtype": image_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["future_speed", "future_steer"],
        },
    }


def convert() -> None:
    args = parse_args()
    import numpy as np

    CommaLeRobotDataset = import_comma_dataset()
    LeRobotDataset = import_lerobot_dataset()

    history_offsets = resolve_history_offsets(args)
    target_size = (args.width, args.height)
    state_dim = 2 + 2 * len(history_offsets)
    dataset_root = args.output_root / args.repo_id

    if dataset_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output dataset already exists at {dataset_root}. "
                "Use --overwrite to recreate it."
            )
        shutil.rmtree(dataset_root)

    features = make_features(
        height=args.height,
        width=args.width,
        state_dim=state_dim,
        use_videos=args.use_videos,
    )

    lerobot_dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_root,
        fps=args.fps,
        robot_type="comma2k19_car",
        features=features,
        use_videos=args.use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
        streaming_encoding=args.streaming_encoding,
    )

    source = CommaLeRobotDataset(
        chunk_path=args.chunk_path,
        target_size=target_size,
        future_time=args.future_time,
        history_offsets=history_offsets,
        mode="episode",
        shuffle_segments=False,
        shuffle_frames=False,
    )

    total_episodes = 0
    total_frames = 0
    try:
        for episode_index, episode in enumerate(source):
            if args.max_episodes is not None and episode_index >= args.max_episodes:
                break

            images = episode["observations"]["image"]
            states = episode["observations"]["state"]
            actions = episode["actions"]
            episode_length = int(episode["length"])
            if args.max_frames_per_episode is not None:
                episode_length = min(episode_length, args.max_frames_per_episode)

            if episode_length <= 0:
                continue

            for frame_idx in range(episode_length):
                frame = {
                    "observation.images.front": tensor_image_to_uint8_hwc(images[frame_idx]),
                    "observation.state": states[frame_idx].detach().cpu().numpy().astype(np.float32),
                    "action": actions[frame_idx].detach().cpu().numpy().astype(np.float32),
                    "task": args.task,
                }
                lerobot_dataset.add_frame(frame)

            lerobot_dataset.save_episode()
            total_episodes += 1
            total_frames += episode_length
            print(
                f"Saved episode {total_episodes} with {episode_length} frames "
                f"({total_frames} total frames)."
            )
    finally:
        lerobot_dataset.finalize()

    print()
    print(f"Conversion complete: {dataset_root}")
    print(f"Repo id: {args.repo_id}")
    print(f"Episodes: {total_episodes}")
    print(f"Frames: {total_frames}")
    print(f"State dim: {state_dim}")
    print(f"History offsets: {history_offsets}")


if __name__ == "__main__":
    convert()
