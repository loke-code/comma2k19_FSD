"""
compute_action_stats.py

Run this once over your full dataset to compute normalization statistics
for the action space [speed (m/s), steer (rad)].

Output:
    Prints hardcoded ACTION_MEAN, ACTION_STD, ACTION_MIN, ACTION_MAX constants
    ready to paste directly into train_smolvla_driving.py
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data_utils.dataloader_VLA import Comma_Segment


def discover_segments(chunk_path: Path):
    segments = []
    for drive in sorted(chunk_path.iterdir()):
        if not drive.is_dir():
            continue
        for seg_path in sorted(
            drive.iterdir(),
            key=lambda x: int(x.name) if x.name.isdigit() else x.name
        ):
            if seg_path.is_dir():
                segments.append(seg_path)
    return segments


def collect_stats(chunk_paths: list[Path], future_window: float = 1.0, num_future_steps: int = 5):
    future_offsets = np.linspace(future_window / num_future_steps, future_window, num_future_steps)

    all_speeds = []
    all_steers = []

    for chunk_path in chunk_paths:
        segments = discover_segments(chunk_path)
        print(f"\nProcessing {chunk_path.name} — {len(segments)} segments found")

        for seg_path in tqdm(segments, desc="Segments"):
            try:
                segment = Comma_Segment(seg_path)
            except Exception as e:
                print(f"  Skipping {seg_path}: {e}")
                continue

            for frame_idx, t_current in enumerate(segment.frame_times):
                if t_current < segment.can_t_min:
                    continue
                if t_current + future_window > segment.can_t_max:
                    continue

                for offset in future_offsets:
                    v_f, alpha_f = segment.get_CAN_data(t_current + offset)
                    all_speeds.append(v_f)
                    all_steers.append(alpha_f)

    return np.array(all_speeds, dtype=np.float32), np.array(all_steers, dtype=np.float32)


def main():
    chunk_paths = [Path("comma2k19_data") / "extracted" / "Chunk_1"]
    future_window = 1.0
    num_future_steps = 5

    print("Action statistics across full dataset")

    speeds, steers = collect_stats(chunk_paths, future_window, num_future_steps)

    action_mean = np.array([speeds.mean(), steers.mean()], dtype=np.float32)
    action_std  = np.array([speeds.std(),  steers.std()],  dtype=np.float32)
    action_min  = np.array([speeds.min(),  steers.min()],  dtype=np.float32)
    action_max  = np.array([speeds.max(),  steers.max()],  dtype=np.float32)

    print("\n" + "="*60)
    print(f"Total action samples collected: {len(speeds):,}")
    print(f"\nSpeed  — mean: {action_mean[0]:.4f}  std: {action_std[0]:.4f}  "
          f"min: {action_min[0]:.4f}  max: {action_max[0]:.4f}")
    print(f"Steer  — mean: {action_mean[1]:.4f}  std: {action_std[1]:.4f}  "
          f"min: {action_min[1]:.4f}  max: {action_max[1]:.4f}")

    print("\n" + "="*60)
    print("="*60)
    print(f"""
        ACTION_MEAN = torch.tensor([{action_mean[0]:.6f}, {action_mean[1]:.6f}], dtype=torch.float32)
        ACTION_STD  = torch.tensor([{action_std[0]:.6f},  {action_std[1]:.6f}],  dtype=torch.float32)
        ACTION_MIN  = torch.tensor([{action_min[0]:.6f}, {action_min[1]:.6f}], dtype=torch.float32)
        ACTION_MAX  = torch.tensor([{action_max[0]:.6f},  {action_max[1]:.6f}],  dtype=torch.float32)
        """)


if __name__ == "__main__":
    main()