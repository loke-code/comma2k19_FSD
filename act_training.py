#!/usr/bin/env python3
"""
End-to-end ACT training pipeline for comma2k19.

Pipeline stages (all can be skipped if already done):
  1. Download + extract Chunks 1-4 via dataset_setup.py
  2. Convert each chunk to LeRobot format via convert_comma2k19_to_lerobot.py
  3. Load the merged LeRobotDataset with observation/action windowing
  4. Train/val split and DataLoader construction
  5. ACT training loop with checkpoint saving
  6. Inference on a saved checkpoint

Usage examples
--------------
# Full pipeline from scratch:
python act_training.py --hf-token hf_xxxx

# Skip download+extract (chunks already on disk), skip conversion:
python act_training.py --skip-download --skip-extract --skip-convert

# Only run inference on a checkpoint:
python act_training.py --skip-download --skip-extract --skip-convert --skip-train \\
    --checkpoint outputs/train/act_comma2k19/checkpoints/010000/pretrained_model

# Smoke-test: 2 episodes per chunk, 64 frames each, 200 training steps:
python act_training.py --max-episodes 2 --max-frames 64 --steps 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- paths ---
    p.add_argument("--base-dir", type=Path, default=Path("./comma2k19_data"),
                   help="Root for raw/extracted comma2k19 data (default: ./comma2k19_data)")
    p.add_argument("--dataset-root", type=Path, default=Path("./lerobot_datasets"),
                   help="Root for LeRobot datasets (default: ./lerobot_datasets)")
    p.add_argument("--repo-id", default="local/comma2k19_act",
                   help="LeRobot dataset repo id (default: local/comma2k19_act)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("./outputs/train/act_comma2k19"),
                   help="Training output / checkpoint directory")
    p.add_argument("--project-scripts-dir", type=Path, default=Path("."),
                   help="Directory containing dataset_setup.py and convert_comma2k19_to_lerobot.py")

    # --- chunks ---
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="Chunk numbers to download and convert (default: 1 2 3 4)")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token for dataset download")

    # --- conversion ---
    p.add_argument("--width",  type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--fps",    type=int, default=20)
    p.add_argument("--future-time", type=float, default=1.0,
                   help="Seconds ahead for the action target (default: 1.0)")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap episodes per chunk (smoke test)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames per episode (smoke test)")

    # --- windowing (for the DataLoader) ---
    p.add_argument("--obs-horizon", type=int, default=1,
                   help="Number of past frames stacked as observation context (default: 1)")
    p.add_argument("--action-horizon", type=int, default=16,
                   help="Number of future action steps in each chunk (default: 16)")
    p.add_argument("--pred-horizon", type=int, default=16,
                   help="Prediction horizon; must be >= action_horizon (default: 16)")

    # --- training ---
    p.add_argument("--device", default="cuda",
                   help="Training device: cuda | mps | cpu (default: cuda)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of episodes reserved for validation (default: 0.1)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    # --- inference ---
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to a pretrained_model directory for inference. "
                        "Defaults to <output-dir>/checkpoints/last/pretrained_model")
    p.add_argument("--inference-episodes", type=int, default=3,
                   help="Number of val episodes to run inference on (default: 3)")

    # --- skip flags ---
    p.add_argument("--skip-download",  action="store_true", help="Skip HF download")
    p.add_argument("--skip-extract",   action="store_true", help="Skip zip extraction")
    p.add_argument("--skip-convert",   action="store_true", help="Skip LeRobot conversion")
    p.add_argument("--skip-train",     action="store_true", help="Skip training loop")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage 1 — download + extract chunks via dataset_setup.py
# ---------------------------------------------------------------------------

def stage_download_extract(args: argparse.Namespace) -> None:
    """Call dataset_setup.py once per chunk."""
    setup_script = args.project_scripts_dir / "dataset_setup.py"
    if not setup_script.exists():
        raise FileNotFoundError(f"dataset_setup.py not found at {setup_script}")

    for chunk in args.chunks:
        print(f"\n{'='*60}")
        print(f"  Stage 1 — Chunk {chunk}: download + extract")
        print(f"{'='*60}")

        cmd = [
            sys.executable, str(setup_script),
            "--chunk", str(chunk),
            "--base-dir", str(args.base_dir),
        ]
        if args.hf_token:
            cmd += ["--hf-token", args.hf_token]
        if args.skip_download:
            cmd += ["--skip-download"]
        if args.skip_extract:
            cmd += ["--skip-extract"]

        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Stage 2 — convert chunks to LeRobot format
# ---------------------------------------------------------------------------

def stage_convert(args: argparse.Namespace) -> None:
    """
    Convert each chunk into the same LeRobot repo-id so all episodes end up
    in one merged dataset. Chunk_1 is converted with --overwrite to start
    fresh; subsequent chunks append by omitting --overwrite.
    """
    convert_script = args.project_scripts_dir / "convert_comma2k19_to_lerobot.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_comma2k19_to_lerobot.py not found at {convert_script}"
        )

    extract_dir = args.base_dir / "extracted"

    for i, chunk in enumerate(args.chunks):
        chunk_path = extract_dir / f"Chunk_{chunk}"
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"Extracted chunk not found: {chunk_path}. "
                "Run without --skip-extract first."
            )

        print(f"\n{'='*60}")
        print(f"  Stage 2 — Chunk {chunk}: convert to LeRobot")
        print(f"{'='*60}")

        cmd = [
            sys.executable, str(convert_script),
            "--chunk-path",  str(chunk_path),
            "--repo-id",     args.repo_id,
            "--output-root", str(args.dataset_root),
            "--width",       str(args.width),
            "--height",      str(args.height),
            "--fps",         str(args.fps),
            "--future-time", str(args.future_time),
        ]
        if args.max_episodes is not None:
            cmd += ["--max-episodes", str(args.max_episodes)]
        if args.max_frames is not None:
            cmd += ["--max-frames-per-episode", str(args.max_frames)]

        # Overwrite only on the first chunk to start clean;
        # subsequent chunks append to the same dataset.
        if i == 0:
            cmd += ["--overwrite"]

        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Stage 3 — load LeRobotDataset with observation + action windowing
# ---------------------------------------------------------------------------

def load_lerobot_dataset(args: argparse.Namespace):
    """
    Load the converted LeRobotDataset.

    Windowing is applied via delta_timestamps:
    - observation.images.front and observation.state: obs_horizon past frames
      at 1/fps spacing, ending at the current frame (index 0).
    - action: pred_horizon future steps at 1/fps spacing starting at t+1.

    This produces tensors of shape:
      observation.images.front : (obs_horizon, H, W, 3)
      observation.state        : (obs_horizon, 2)
      action                   : (pred_horizon, 2)

    ACT training then uses action[:action_horizon] as the chunk target.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset_path = args.dataset_root / args.repo_id
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"LeRobot dataset not found at {dataset_path}. "
            "Run without --skip-convert first."
        )

    dt = 1.0 / args.fps  # seconds per frame

    # Observation window: obs_horizon frames ending at t=0
    obs_offsets = [round(-dt * (args.obs_horizon - 1 - k), 6)
                   for k in range(args.obs_horizon)]

    # Action / prediction window: pred_horizon steps starting at t+dt
    action_offsets = [round(dt * (k + 1), 6) for k in range(args.pred_horizon)]

    delta_timestamps = {
        "observation.images.front": obs_offsets,
        "observation.state":        obs_offsets,
        "action":                   action_offsets,
    }

    print("\nLoading LeRobotDataset with windowing:")
    print(f"  obs_horizon   : {args.obs_horizon}  offsets: {obs_offsets}")
    print(f"  pred_horizon  : {args.pred_horizon}  offsets: {action_offsets}")
    print(f"  action_horizon: {args.action_horizon}  (first {args.action_horizon} of pred window used)")

    dataset = LeRobotDataset(
        args.repo_id,
        root=dataset_path,
        delta_timestamps=delta_timestamps,
    )

    print(f"  Total frames  : {len(dataset)}")
    print(f"  Episodes      : {dataset.num_episodes}")
    return dataset


# ---------------------------------------------------------------------------
# Stage 4 — train / val split
# ---------------------------------------------------------------------------

def split_dataset(dataset, val_fraction: float, seed: int):
    """
    Episode-level train/val split so that no episode straddles the boundary.
    Returns (train_dataset, val_dataset) as Subset objects.
    """
    import random
    from torch.utils.data import Subset

    random.seed(seed)

    num_episodes = dataset.num_episodes
    episode_ids  = list(range(num_episodes))
    random.shuffle(episode_ids)

    n_val = max(1, int(num_episodes * val_fraction))
    val_episode_ids  = set(episode_ids[:n_val])
    train_episode_ids = set(episode_ids[n_val:])

    # Map episode ids to flat frame indices using episode_data_index
    # LeRobotDataset stores per-episode start/end under episode_data_index
    train_indices: list[int] = []
    val_indices:   list[int] = []

    ep_index = dataset.episode_data_index  # dict with "from" and "to" tensors
    for ep_id in range(num_episodes):
        start = int(ep_index["from"][ep_id])
        end   = int(ep_index["to"][ep_id])
        indices = list(range(start, end))
        if ep_id in val_episode_ids:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    print(f"\nTrain/val split (seed={seed}, val_fraction={val_fraction}):")
    print(f"  Train episodes: {len(train_episode_ids)}  frames: {len(train_indices)}")
    print(f"  Val   episodes: {len(val_episode_ids)}   frames: {len(val_indices)}")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        sorted(val_episode_ids),
    )


def make_dataloaders(train_set, val_set, args: argparse.Namespace):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Stage 5 — ACT training loop
# ---------------------------------------------------------------------------

def make_act_policy(dataset, args: argparse.Namespace):
    """
    Instantiate ACT policy from LeRobot, inferring obs/action dims from the
    dataset features.

    observation.state shape : (obs_horizon, 2)   → state_dim = obs_horizon * 2
    action shape             : (pred_horizon, 2)  → action_dim = 2, chunk = action_horizon
    """
    import torch

    try:
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as exc:
        raise SystemExit(
            "Could not import ACTPolicy. Make sure lerobot is installed: pip install lerobot"
        ) from exc

    # Build config aligned with the windowed dataset schema
    cfg = ACTConfig(
        input_shapes={
            "observation.images.front": [args.obs_horizon, args.height, args.width, 3],
            "observation.state":        [args.obs_horizon, 2],
        },
        output_shapes={
            "action": [args.action_horizon, 2],
        },
        chunk_size=args.action_horizon,
        n_action_steps=args.action_horizon,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda"
                          else "cpu")
    policy = ACTPolicy(cfg, dataset_stats=dataset.stats)
    policy.to(device)

    print(f"\nACT policy on {device}")
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {total_params:,}")

    return policy, device


def train(policy, train_loader, val_loader, args: argparse.Namespace, device) -> None:
    import torch

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_iter = iter(train_loader)
    policy.train()

    print(f"\nTraining for {args.steps} steps ...")

    for step in range(1, args.steps + 1):
        # Refill iterator when exhausted
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

        optimizer.zero_grad()
        loss, info = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            val_loss = evaluate(policy, val_loader, device, max_batches=20)
            print(
                f"  step {step:>6}/{args.steps} | "
                f"train_loss: {loss.item():.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e}"
            )
            policy.train()

        if step % args.save_every == 0 or step == args.steps:
            ckpt_dir = output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
            save_checkpoint(policy, ckpt_dir, step)
            # Always keep a symlink/copy called "last"
            last_dir = output_dir / "checkpoints" / "last" / "pretrained_model"
            save_checkpoint(policy, last_dir, step)

    print("Training complete.")


def evaluate(policy, val_loader, device, max_batches: int = None) -> float:
    """Run one pass over val_loader and return mean loss."""
    import torch

    policy.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def save_checkpoint(policy, ckpt_dir: Path, step: int) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(ckpt_dir))
    print(f"  Checkpoint saved → {ckpt_dir}")


# ---------------------------------------------------------------------------
# Stage 6 — inference on a checkpoint
# ---------------------------------------------------------------------------

def load_policy_from_checkpoint(ckpt_dir: Path, device):
    """Load an ACTPolicy from a saved pretrained_model directory."""
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as exc:
        raise SystemExit("Could not import ACTPolicy.") from exc

    print(f"\nLoading checkpoint from {ckpt_dir}")
    policy = ACTPolicy.from_pretrained(str(ckpt_dir))
    policy.to(device)
    policy.eval()
    return policy


def run_inference(policy, val_set, val_episode_ids: list[int],
                  dataset, args: argparse.Namespace, device) -> None:
    """
    Run inference on the first --inference-episodes validation episodes.

    For each episode we iterate its frames in order, feed (image, state) to
    the policy, and collect predicted vs ground-truth action chunks.

    Output: per-episode mean absolute error on speed and steer.
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    n_to_run = min(args.inference_episodes, len(val_episode_ids))
    print(f"\nInference on {n_to_run} validation episode(s) ...")

    ep_index = dataset.episode_data_index

    for ep_num, ep_id in enumerate(val_episode_ids[:n_to_run]):
        start  = int(ep_index["from"][ep_id])
        end    = int(ep_index["to"][ep_id])
        ep_set = Subset(val_set.dataset, list(range(start, end)))
        loader = DataLoader(ep_set, batch_size=1, shuffle=False)

        pred_speeds, gt_speeds   = [], []
        pred_steers, gt_steers   = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if hasattr(v, "to") else v
                         for k, v in batch.items()}

                # policy.select_action returns the action chunk (action_horizon, 2)
                # We take the first step of the chunk as the immediate prediction.
                action_chunk = policy.select_action(batch)   # (1, action_horizon, 2)
                pred_action  = action_chunk[0, 0]             # (2,) first step

                gt_action = batch["action"][0, 0]             # (2,) first future step

                pred_speeds.append(pred_action[0].item())
                pred_steers.append(pred_action[1].item())
                gt_speeds.append(gt_action[0].item())
                gt_steers.append(gt_action[1].item())

        pred_speeds = np.array(pred_speeds)
        pred_steers = np.array(pred_steers)
        gt_speeds   = np.array(gt_speeds)
        gt_steers   = np.array(gt_steers)

        mae_speed = np.mean(np.abs(pred_speeds - gt_speeds))
        mae_steer = np.mean(np.abs(pred_steers - gt_steers))

        print(f"  Episode {ep_id:>3} ({end - start} frames) | "
              f"MAE speed: {mae_speed:.4f} m/s | "
              f"MAE steer: {mae_steer:.4f} deg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    import torch
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Stage 1: download + extract
    # ------------------------------------------------------------------
    if not args.skip_download or not args.skip_extract:
        stage_download_extract(args)
    else:
        print("Skipping download and extraction.")

    # ------------------------------------------------------------------
    # Stage 2: convert to LeRobot
    # ------------------------------------------------------------------
    if not args.skip_convert:
        stage_convert(args)
    else:
        print("Skipping LeRobot conversion.")

    # ------------------------------------------------------------------
    # Stage 3: load dataset with windowing
    # ------------------------------------------------------------------
    dataset = load_lerobot_dataset(args)

    # ------------------------------------------------------------------
    # Stage 4: train/val split + dataloaders
    # ------------------------------------------------------------------
    train_set, val_set, val_episode_ids = split_dataset(
        dataset, args.val_fraction, args.seed
    )
    train_loader, val_loader = make_dataloaders(train_set, val_set, args)

    # ------------------------------------------------------------------
    # Stage 5: train
    # ------------------------------------------------------------------
    if not args.skip_train:
        policy, device = make_act_policy(dataset, args)
        train(policy, train_loader, val_loader, args, device)
    else:
        print("Skipping training.")
        device = torch.device(
            args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
        )

    # ------------------------------------------------------------------
    # Stage 6: inference
    # ------------------------------------------------------------------
    ckpt_dir = args.checkpoint
    if ckpt_dir is None:
        ckpt_dir = args.output_dir / "checkpoints" / "last" / "pretrained_model"

    if not ckpt_dir.exists():
        print(f"\nNo checkpoint found at {ckpt_dir} — skipping inference.")
        return

    infer_policy = load_policy_from_checkpoint(ckpt_dir, device)
    run_inference(infer_policy, val_set, val_episode_ids, dataset, args, device)


if __name__ == "__main__":
    main()
