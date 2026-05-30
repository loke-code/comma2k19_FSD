#!/usr/bin/env python3
"""
Train an ACT policy on a materialized comma2k19 LeRobotDataset.

This script is a thin, reproducible wrapper around LeRobot's official training
CLI. ACT adapts its state/action/camera dimensions from the saved
LeRobotDataset metadata, so the recommended workflow is:

1. Convert comma2k19 with convert_comma2k19_to_lerobot.py.
2. Train with this script.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-repo-id",
        default="local/comma2k19_act",
        help="Repo id used when the local LeRobotDataset was created.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("lerobot_datasets/local/comma2k19_act"),
        help="Local root containing the materialized LeRobotDataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/act_comma2k19"),
        help="Directory for ACT checkpoints and logs.",
    )
    parser.add_argument("--job-name", default="act_comma2k19", help="LeRobot training job name.")
    parser.add_argument("--device", default="cuda", help="Training device: cuda, cpu, or mps.")
    parser.add_argument("--batch-size", type=int, default=8, help="ACT training batch size.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps.")
    parser.add_argument("--num-workers", type=int, default=4, help="PyTorch dataloader workers.")
    parser.add_argument(
        "--policy-repo-id",
        default=None,
        help="Optional Hugging Face model repo id for the trained policy.",
    )
    parser.add_argument(
        "--wandb-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Push trained policy to the Hugging Face Hub if supported by your LeRobot version.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the lerobot-train command without executing it.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional raw arguments appended to lerobot-train. Prefix with --.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    executable = shutil.which("lerobot-train")
    if executable is None:
        raise SystemExit(
            "Could not find lerobot-train on PATH. Install LeRobot in this environment first."
        )

    if not args.dataset_root.exists():
        raise SystemExit(
            f"Dataset root does not exist: {args.dataset_root}\n"
            "Run convert_comma2k19_to_lerobot.py first."
        )

    command = [
        executable,
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={args.dataset_root}",
        "--policy.type=act",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--num_workers={args.num_workers}",
        f"--wandb.enable={str(args.wandb_enable).lower()}",
        f"--policy.push_to_hub={str(args.push_to_hub).lower()}",
    ]

    if args.policy_repo_id:
        command.append(f"--policy.repo_id={args.policy_repo_id}")

    if args.extra_args:
        command.extend(args.extra_args)

    return command


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print("Running ACT training command:")
    print(" ".join(str(part) for part in command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
