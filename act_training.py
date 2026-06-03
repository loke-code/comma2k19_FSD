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
from ast import List
import subprocess
import sys
from pathlib import Path
import numpy as np

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
    p.add_argument("--skip-eval",      action="store_true", help="Skip evaluation with lerobot-eval")

    p.add_argument("--eval-batch-size", type=int, default=10,
                   help="Batch size for lerobot-eval vectorized environments")
    p.add_argument("--eval-n-episodes", type=int, default=10,
                   help="Number of episodes to evaluate with lerobot-eval")
    p.add_argument("--eval-env-type", type=str, default="pusht",
                  help="Environment type for lerobot-eval (NOTE: comma2k19 is offline data, use eval_and_plot() in notebook instead)")
    p.add_argument("--eval-use-amp", action="store_true",
                   help="Enable AMP during evaluation")
    p.add_argument("--eval-output-dir", type=Path, default=None,
                   help="Output directory for evaluation results (default: <output-dir>/eval)")

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

def import_lerobot_dataset_cls():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def make_delta_timestamps(fps: int, obs_horizon: int, pred_horizon: int) -> dict:
    """
    Build the delta_timestamps dict that tells LeRobotDataset how to window
    each sample.

    obs_horizon  past frames (including current) for image and state:
      e.g. obs_horizon=1, fps=20 → offsets=[0.0]   (current frame only)
      e.g. obs_horizon=3, fps=20 → offsets=[-0.1, -0.05, 0.0]

    pred_horizon future frames for action:
      e.g. pred_horizon=16, fps=20 → offsets=[0.05, 0.1, ..., 0.8]

    Resulting tensor shapes per sample:
      observation.images.front : (obs_horizon, H, W, 3)
      observation.state        : (obs_horizon, 2)
      action                   : (pred_horizon, 2)
    """
    dt = 1.0 / fps
    obs_offsets    = [round(-dt * (obs_horizon - 1 - k), 6) for k in range(obs_horizon)]
    action_offsets = [round(dt * (k + 1), 6) for k in range(pred_horizon)]
    return {
        "observation.images.front": obs_offsets,
        "observation.state":        obs_offsets,
        "action":                   action_offsets,
    }


def load_single_split(
    repo_id: str,
    dataset_root: Path,
    delta_timestamps: dict,
    episodes: List,
):
    """
    Load one LeRobotDataset for a given split string.

    split follows HuggingFace convention, e.g.:
      "train[:90%]"  — first 90% of episodes
      "train[90%:]"  — last  10% of episodes

    root must be the PARENT directory of repo_id so LeRobot can find
    meta/info.json locally without hitting the Hub.
    e.g.  repo_id="local/comma2k19_act", root="./lerobot_datasets"
      → looks for ./lerobot_datasets/local/comma2k19_act/meta/info.json
    """
    LeRobotDataset = import_lerobot_dataset_cls()
    return LeRobotDataset(
        repo_id=repo_id,
        root=str(dataset_root / repo_id),
        delta_timestamps=delta_timestamps,
        episodes=episodes,
        video_backend=None
    )


def split_dataset(
    repo_id: str,
    dataset_root,
    delta_timestamps: dict,
    val_fraction: float,
):
    """
    Create train/val datasets by splitting episode IDs.
    """

    # Load once to discover episode count
    full_dataset = load_single_split(
        repo_id,
        dataset_root,
        delta_timestamps,
        episodes=None,
    )

    num_episodes = full_dataset.num_episodes

    episode_ids = np.arange(num_episodes)

    # reproducible shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(episode_ids)

    num_val = max(1, int(num_episodes * val_fraction))

    val_episodes = episode_ids[:num_val].tolist()
    train_episodes = episode_ids[num_val:].tolist()

    train_dataset = load_single_split(
        repo_id,
        dataset_root,
        delta_timestamps,
        episodes=train_episodes,
    )

    val_dataset = load_single_split(
        repo_id,
        dataset_root,
        delta_timestamps,
        episodes=val_episodes,
    )

    print(
        f"  Train — episodes: {train_dataset.num_episodes}  frames: {len(train_dataset)}"
    )
    print(
        f"  Val   — episodes: {val_dataset.num_episodes}  frames: {len(val_dataset)}"
    )

    return train_dataset, val_dataset

def load_lerobot_dataset(args: argparse.Namespace):
    """
    Orchestrates dataset loading for the training pipeline.
    Validates the dataset path, builds delta_timestamps, then delegates
    to split_dataset for the actual LeRobotDataset construction.
    """
    dataset_path = args.dataset_root / args.repo_id
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"LeRobot dataset not found at {dataset_path}. "
            "Run without --skip-convert first."
        )

    delta_timestamps = make_delta_timestamps(args.fps, args.obs_horizon, args.pred_horizon)

    print("\nDataset windowing:")
    print(f"  obs_horizon   : {args.obs_horizon}")
    print(f"  pred_horizon  : {args.pred_horizon}")
    print(f"  action_horizon: {args.action_horizon}")
    print(f"  delta_timestamps: {delta_timestamps}")

    return split_dataset(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        val_fraction=args.val_fraction,
    )


# ---------------------------------------------------------------------------
# Stage 4 — train via lerobot-train CLI
# ---------------------------------------------------------------------------

def stage_train(args: argparse.Namespace) -> None:
    """
    Delegate training entirely to LeRobot's own lerobot-train CLI.

    lerobot-train handles:
      - ACTConfig / ACTPolicy construction from the dataset's feature schema
      - DataLoader creation (batch size, workers, sampler)
      - AdamW optimiser + cosine LR schedule
      - Grad clipping, EMA, WandB/tensorboard logging
      - Checkpoint saving (every --save_freq steps + final)
      - Resuming from the last checkpoint automatically

    We only pass the knobs that differ from LeRobot defaults:
      policy=act                     use ACT architecture
      dataset_repo_id / dataset_root point at the local converted dataset
      training.num_workers           forwarded from --num-workers
      training.batch_size            forwarded from --batch-size
      training.num_train_steps       forwarded from --steps
      training.save_freq             forwarded from --save-every
      training.log_freq              forwarded from --log-every
      training.eval_freq             same as log_every
      device                         forwarded from --device
      output_dir                     forwarded from --output-dir
      train_split / val_split        derived from --val-fraction
    """
    val_pct   = int(args.val_fraction * 100)
    train_pct = 100 - val_pct

    cmd = [
    "lerobot-train",
    "--policy.type", "act",
    "--policy.push_to_hub","False", 
    "--dataset.repo_id", args.repo_id,
    "--dataset.root", str(args.dataset_root / args.repo_id),
    "--batch_size", str(args.batch_size),
    "--num_workers", str(args.num_workers),
    "--steps", str(args.steps),
    "--save_freq", str(args.save_every),
    "--log_freq", str(args.log_every),
    "--eval_freq", str(args.log_every),
    "--output_dir", str(args.output_dir),
    # "--dataset.video_backend", args.video_backend

    ]

    print("\nLaunching LeRobot training:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def stage_eval(args: argparse.Namespace) -> None:
    """
    Evaluate a trained checkpoint using LeRobot's lerobot-eval CLI.

    If no checkpoint path is provided, the default is
    <output_dir>/checkpoints/last/pretrained_model.
    """
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = args.output_dir / "checkpoints" / "last" / "pretrained_model"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {checkpoint_path}. "
            "Provide --checkpoint or run training first."
        )

    eval_output_dir = args.eval_output_dir or (args.output_dir / "eval")
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lerobot-eval",
        "--policy.pretrained_path", str(checkpoint_path),
        "--env.type", args.eval_env_type,
        "--eval.batch_size", str(args.eval_batch_size),
        "--eval.n_episodes", str(args.eval_n_episodes),
        "--policy.use_amp", "true" if args.eval_use_amp else "false",
        "--policy.device", args.device,
        "--output_dir", str(eval_output_dir),
    ]

    print("\nLaunching LeRobot evaluation:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Stage 6 — inference on a checkpoint
# ---------------------------------------------------------------------------

def load_policy_from_checkpoint(ckpt_dir: Path, device):
    """
    Load an ACTPolicy from a pretrained_model directory saved by lerobot-train.
    Uses LeRobot's own from_pretrained so the config is restored exactly as saved.
    """
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as exc:
        raise SystemExit("Could not import ACTPolicy.") from exc

    print(f"\nLoading checkpoint from {ckpt_dir}")
    policy = ACTPolicy.from_pretrained(str(ckpt_dir))
    policy.to(device)
    policy.eval()
    return policy


def run_inference(policy, val_dataset, args: argparse.Namespace, device) -> None:
    """
    Offline inference over val episodes.

    How ACTPolicy.select_action works
    ----------------------------------
    ACT generates a chunk of `chunk_size` actions in one forward pass, then
    serves them one at a time from an internal queue on subsequent calls.
    This means:
      - Call policy.reset() at the start of every episode to flush the queue.
      - Pass ONE frame at a time as an observation dict (no batch dimension on
        the sequence axis — LeRobot adds the batch dim internally).
      - select_action returns a single (action_dim,) tensor, i.e. (2,) here.
        Do NOT index it as [0, 0]; it is already the scalar action for this step.

    We compare the first action dimension (speed) and second (steer) against
    the ground-truth first future step stored in batch["action"][:, 0, :].
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    n_to_run = min(args.inference_episodes, val_dataset.num_episodes)
    print(f"\nInference on {n_to_run} of {val_dataset.num_episodes} val episode(s) ...")

    ep_index = val_dataset.episode_data_index

    for ep_num in range(n_to_run):
        start = int(ep_index["from"][ep_num])
        end   = int(ep_index["to"][ep_num])
        ep_set = Subset(val_dataset, list(range(start, end)))
        loader = DataLoader(ep_set, batch_size=1, shuffle=False)

        # Reset the action chunk queue at the start of each episode
        policy.reset()

        pred_speeds, gt_speeds = [], []
        pred_steers, gt_steers = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if hasattr(v, "to") else v
                         for k, v in batch.items()}

                # select_action consumes one step from the internal chunk buffer,
                # re-running the encoder only when the buffer is empty.
                # Returns: (2,) — [speed, steer] for this timestep.
                pred_action = policy.select_action(batch)  # (2,)

                # Ground truth: first future step from the action window
                # batch["action"] shape: (1, pred_horizon, 2)
                gt_action = batch["action"][0, 0]          # (2,)

                pred_speeds.append(pred_action[0].item())
                pred_steers.append(pred_action[1].item())
                gt_speeds.append(gt_action[0].item())
                gt_steers.append(gt_action[1].item())

        mae_speed = float(np.mean(np.abs(np.array(pred_speeds) - np.array(gt_speeds))))
        mae_steer = float(np.mean(np.abs(np.array(pred_steers) - np.array(gt_steers))))

        print(f"  Val ep {ep_num:>3}  ({end - start} frames) | "
              f"MAE speed: {mae_speed:.4f} m/s | "
              f"MAE steer: {mae_steer:.4f} deg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # LeRobot's ACT policy currently only supports an observation horizon of 1
    if args.obs_horizon != 1:
        raise ValueError(
            f"LeRobot's ACT policy currently only supports an observation horizon of 1. "
            f"You provided --obs-horizon={args.obs_horizon}. Please run with --obs-horizon=1."
        )

    import torch

    # ------------------------------------------------------------------
    # Stage 1: download + extract chunks 1-4
    # ------------------------------------------------------------------
    if not args.skip_download or not args.skip_extract:
        stage_download_extract(args)
    else:
        print("Skipping download and extraction.")

    # ------------------------------------------------------------------
    # Stage 2: convert each chunk to LeRobot format
    # ------------------------------------------------------------------
    if not args.skip_convert:
        stage_convert(args)
    else:
        print("Skipping LeRobot conversion.")

    # ------------------------------------------------------------------
    # Stage 3: train via lerobot-train (handles dataloaders, optimiser,
    #           checkpointing — no hand-written loop needed)
    # ------------------------------------------------------------------
    if not args.skip_train:
        stage_train(args)
    else:
        print("Skipping training.")

    if not args.skip_eval:
        stage_eval(args)
    else:
        print("Skipping evaluation.")

    # ------------------------------------------------------------------
    # Stage 4: inference on a checkpoint
    # ------------------------------------------------------------------
    ckpt_dir = args.checkpoint
    if ckpt_dir is None:
        ckpt_dir = args.output_dir / "checkpoints" / "last" / "pretrained_model"

    if not ckpt_dir.exists():
        print(f"\nNo checkpoint found at {ckpt_dir} — skipping inference.")
        return

    # Load val dataset (needed for inference loop)
    _, val_dataset = load_lerobot_dataset(args)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    infer_policy = load_policy_from_checkpoint(ckpt_dir, device)
    run_inference(infer_policy, val_dataset, args, device)


if __name__ == "__main__":
    main()