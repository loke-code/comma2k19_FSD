#!/usr/bin/env python3
"""
Convert comma2k19 extracted chunks into a local LeRobotDataset for ACT training.

One frame per timestep. No CAN history. No windowing.

Each LeRobot frame contains:
  observation.images.front  (H, W, 3)  uint8 RGB
  observation.state         (2,)        [speed_mps, steer_deg]  at time t
  action                    (2,)        [speed_mps, steer_deg]  at time t + future_time

Each comma2k19 segment becomes one LeRobot episode.
Frames where t + future_time exceeds CAN coverage are dropped.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_TASK = "Predict future speed and steering from the current driving scene."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunk-path", type=Path, required=True,
                   help="Path to extracted comma2k19 chunk, e.g. comma2k19_data/extracted/Chunk_1")
    p.add_argument("--repo-id", default="local/comma2k19_act",
                   help="LeRobot dataset repo id (default: local/comma2k19_act)")
    p.add_argument("--output-root", type=Path, default=Path("lerobot_datasets"),
                   help="Root directory for the output dataset")
    p.add_argument("--width",  type=int, default=256, help="Output image width  (default: 256)")
    p.add_argument("--height", type=int, default=256, help="Output image height (default: 256)")
    p.add_argument("--fps",    type=int, default=20,  help="Dataset FPS (default: 20)")
    p.add_argument("--future-time", type=float, default=1.0,
                   help="Seconds ahead for the action target (default: 1.0)")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap number of episodes for smoke tests")
    p.add_argument("--max-frames-per-episode", type=int, default=None,
                   help="Cap frames per episode for smoke tests")
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="Task string stored with every frame")
    p.add_argument("--use-videos", action=argparse.BooleanOptionalAction, default=False,
                   help="Store images as video (default: False)")
    p.add_argument("--streaming-encoding", action="store_true",
                   help="Use LeRobot streaming video encoding")
    p.add_argument("--batch-encoding-size", type=int, default=1)
    p.add_argument("--image-writer-threads",   type=int, default=4)
    p.add_argument("--image-writer-processes", type=int, default=0)
    p.add_argument("--overwrite", action="store_true",
                   help="Delete existing output dataset before conversion")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise SystemExit("LeRobot not installed. Run: pip install lerobot") from exc
    return LeRobotDataset


def import_comma_segment():
    try:
        from data_utils.data_loader import Comma_Segment
    except ImportError as exc:
        raise SystemExit(
            "Could not import Comma_Segment. Run from the comma2k19_FSD project root "
            "or add it to PYTHONPATH.\n"
            f"  {exc}"
        ) from exc
    return Comma_Segment


# ---------------------------------------------------------------------------
# Dataset feature schema
# ---------------------------------------------------------------------------

def make_features(height: int, width: int, use_videos: bool) -> dict[str, dict[str, Any]]:
    """
    Fixed schema: one image, one (speed, steer) state, one (speed, steer) action.

    observation.state is shape (2,) — just the current speed and steer.
    There is no history dimension and no temporal token axis.
    The state feeds directly into the ACT encoder as a plain 2-D vector.
    """
    return {
        "observation.images.front": {
            "dtype": "video" if use_videos else "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["speed_mps", "steer_deg"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["future_speed_mps", "future_steer_deg"],
        },
    }


# ---------------------------------------------------------------------------
# Segment discovery
# ---------------------------------------------------------------------------

def discover_segments(chunk_path: Path) -> list[Path]:
    """Walk chunk → drive dirs → segment dirs, sorted numerically."""
    segments: list[Path] = []
    for drive in sorted(chunk_path.iterdir()):
        if not drive.is_dir():
            continue
        for seg in sorted(drive.iterdir(),
                          key=lambda x: int(x.name) if x.name.isdigit() else x.name):
            if seg.is_dir():
                segments.append(seg)
    return segments


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    """Center-crop to square, resize to target_wh, BGR → RGB. Returns (H, W, 3) uint8."""
    h, w = frame.shape[:2]
    start_x = (w - h) // 2
    frame = frame[:, start_x: start_x + h]
    frame = cv2.resize(frame, target_wh)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------

def convert() -> None:
    args = parse_args()

    Comma_Segment  = import_comma_segment()
    LeRobotDataset = import_lerobot_dataset()

    target_wh    = (args.width, args.height)   # (W, H) for cv2.resize
    dataset_root = args.output_root / args.repo_id

    if dataset_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output dataset already exists at {dataset_root}. "
                "Use --overwrite to recreate it."
            )
        shutil.rmtree(dataset_root)

    lerobot_dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_root,
        fps=args.fps,
        robot_type="comma2k19_car",
        features=make_features(args.height, args.width, args.use_videos),
        use_videos=args.use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
        streaming_encoding=args.streaming_encoding,
    )

    segment_paths = discover_segments(args.chunk_path)
    if not segment_paths:
        raise SystemExit(f"No segments found under {args.chunk_path}")

    print(f"Found {len(segment_paths)} segments.")
    print(f"State : [speed_mps, steer_deg] at t          shape (2,)")
    print(f"Action: [speed_mps, steer_deg] at t + {args.future_time}s  shape (2,)")
    print()

    total_episodes = 0
    total_frames   = 0

    try:
        for seg_idx, seg_path in enumerate(segment_paths):
            if args.max_episodes is not None and total_episodes >= args.max_episodes:
                break

            print(f"[{seg_idx + 1}/{len(segment_paths)}] {seg_path}")

            # Load segment metadata (frame times, CAN arrays)
            segment = Comma_Segment(seg_path, target_size=target_wh)
            max_can_t = min(segment.speed_t[-1], segment.steer_t[-1])

            # Decode all frames for this segment
            cap = cv2.VideoCapture(segment.video_path)
            raw_frames: list[np.ndarray] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                raw_frames.append(preprocess_frame(frame, target_wh))
            cap.release()

            n_frames = min(len(raw_frames), len(segment.frame_times))
            if args.max_frames_per_episode is not None:
                n_frames = min(n_frames, args.max_frames_per_episode)

            episode_frames = 0
            for i in range(n_frames):
                t_current = float(segment.frame_times[i])
                t_future  = t_current + args.future_time

                # Drop frames where the future target falls outside CAN coverage
                if t_future > max_can_t:
                    break

                speed_now,  steer_now  = segment.get_CAN_data(t_current)
                speed_then, steer_then = segment.get_CAN_data(t_future)

                lerobot_dataset.add_frame({
                    "observation.images.front": raw_frames[i],
                    "observation.state": np.array([speed_now,  steer_now],  dtype=np.float32),
                    "action":            np.array([speed_then, steer_then], dtype=np.float32),
                    "task": args.task,
                })
                episode_frames += 1

            del raw_frames  # free video memory before next segment

            if episode_frames == 0:
                print("  Skipping — no valid frames.")
                continue

            lerobot_dataset.save_episode()
            total_episodes += 1
            total_frames   += episode_frames
            print(f"  {episode_frames} frames  ({total_frames} total)")

    finally:
        lerobot_dataset.finalize()

    print()
    print("=" * 55)
    print(f"Done.   Dataset : {dataset_root}")
    print(f"        Repo id : {args.repo_id}")
    print(f"       Episodes : {total_episodes}")
    print(f"         Frames : {total_frames}")
    print("=" * 55)


if __name__ == "__main__":
    convert()
