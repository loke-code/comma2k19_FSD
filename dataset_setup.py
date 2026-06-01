#!/usr/bin/env python3
"""
comma2k19 dataset setup: download, extract, and verify a chunk from HuggingFace.

Usage examples
--------------
# Download and extract Chunk_1 (default):
python dataset_setup.py

# Download Chunk_3 with an HF token:
python dataset_setup.py --chunk 3 --hf-token hf_xxxx

# Skip download if already present, just extract and verify:
python dataset_setup.py --skip-download

# Only verify an already-extracted chunk:
python dataset_setup.py --skip-download --skip-extract

# Change base directory:
python dataset_setup.py --base-dir /data/comma2k19 --chunk 2
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunk", type=int, default=1, choices=range(1, 11), metavar="N",
                   help="Chunk number to download (1–10, default: 1)")
    p.add_argument("--base-dir", type=Path, default=Path("./comma2k19_data"),
                   help="Root directory for raw and extracted data (default: ./comma2k19_data)")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token for private/gated access (optional)")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download — use an already-present zip file")
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip extraction — use an already-extracted directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_chunk(chunk: int, base_dir: Path, hf_token: str | None) -> Path:
    """Download Chunk_N.zip from HuggingFace and return its local path."""
    raw_dir = base_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    filename = f"raw_data/Chunk_{chunk}.zip"
    print(f"Downloading {filename} from commaai/comma2k19 ...")

    file_path = hf_hub_download(
        repo_id="commaai/comma2k19",
        filename=filename,
        repo_type="dataset",
        local_dir=base_dir,
        local_dir_use_symlinks=False,
        token=hf_token,
    )

    size_gb = os.path.getsize(file_path) / (1024 ** 3)
    print(f"Downloaded: {file_path}  ({size_gb:.2f} GB)")
    if size_gb < 0.1:
        raise RuntimeError(f"File is suspiciously small ({size_gb:.3f} GB) — download may have failed.")

    return Path(file_path)


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract_chunk(zip_path: Path, extract_dir: Path) -> Path:
    """
    Extract zip_path into extract_dir, sanitising '|' characters in member names.
    Returns the path to the extracted chunk directory.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_members = zf.namelist()
        files = [m for m in all_members if not m.endswith("/")]
        print(f"Found {len(files)} files in {zip_path.name}. Extracting ...")

        for member in tqdm(files, desc="Extracting"):
            clean_name = member.replace("|", "_")
            target = extract_dir / clean_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

    print(f"Extraction complete: {extract_dir.resolve()}")
    return extract_dir


# ---------------------------------------------------------------------------
# CAN data loader
# ---------------------------------------------------------------------------

def load_comma_log(file_path: Path) -> np.ndarray:
    """Load a comma2k19 raw float64 log file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    try:
        return np.load(file_path)
    except Exception:
        return np.fromfile(file_path, dtype=np.float64)


# ---------------------------------------------------------------------------
# Verification passes
# ---------------------------------------------------------------------------

def verify_can_values(chunk_path: Path) -> None:
    """Print min/max speed and steer values for every segment in the chunk."""
    print(f"\n{'='*65}")
    print("CAN value ranges")
    print(f"{'='*65}")

    for drive in sorted(chunk_path.iterdir()):
        if not drive.is_dir():
            continue
        print(f"\n--- Drive: {drive.name} ---")

        segments = sorted(
            [s for s in drive.iterdir() if s.is_dir()],
            key=lambda x: int(x.name) if x.name.isdigit() else x.name,
        )

        for seg in segments:
            steer_path = seg / "processed_log" / "CAN" / "steering_angle" / "value"
            speed_path = seg / "processed_log" / "CAN" / "speed" / "value"
            try:
                steer = load_comma_log(steer_path)
                speed = load_comma_log(speed_path)
                print(
                    f"  Seg {seg.name:>2} | "
                    f"Steer [{np.min(steer):7.2f}, {np.max(steer):7.2f}] | "
                    f"Speed [{np.min(speed):6.2f}, {np.max(speed):6.2f}] | "
                    f"Points [speed={len(speed)}, steer={len(steer)}]"
                )
            except FileNotFoundError:
                print(f"  Seg {seg.name:>2} | [!] Missing CAN log files")
            except Exception as e:
                print(f"  Seg {seg.name:>2} | [!] Error: {e}")


def verify_can_sampling(chunk_path: Path) -> None:
    """Print CAN sampling frequency and timestamp jitter for every segment."""
    print(f"\n{'='*65}")
    print("CAN sampling rates")
    print(f"{'='*65}")

    for drive in sorted(chunk_path.iterdir()):
        if not drive.is_dir():
            continue
        print(f"\nDrive: {drive.name}")

        segments = sorted(
            [s for s in drive.iterdir() if s.is_dir()],
            key=lambda x: int(x.name) if x.name.isdigit() else x.name,
        )

        for seg in segments:
            steer_t_path = seg / "processed_log" / "CAN" / "steering_angle" / "t"
            speed_t_path = seg / "processed_log" / "CAN" / "speed" / "t"
            try:
                s_t = load_comma_log(steer_t_path)
                v_t = load_comma_log(speed_t_path)

                s_diff = np.diff(s_t)
                v_diff = np.diff(v_t)
                s_mean = np.mean(s_diff)
                v_mean = np.mean(v_diff)

                print(
                    f"  Seg {seg.name:>2} | "
                    f"Steer: {(1/s_mean if s_mean > 0 else 0):5.1f} Hz "
                    f"(avg {s_mean:.4f}s, ±{np.std(s_diff):.5f}) | "
                    f"Speed: {(1/v_mean if v_mean > 0 else 0):5.1f} Hz "
                    f"(avg {v_mean:.4f}s, ±{np.std(v_diff):.5f})"
                )
            except FileNotFoundError:
                print(f"  Seg {seg.name:>2} | [!] Missing timestamp files")
            except Exception as e:
                print(f"  Seg {seg.name:>2} | [!] Error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    base_dir    = args.base_dir
    raw_dir     = base_dir / "raw_data"
    extract_dir = base_dir / "extracted"
    zip_path    = raw_dir / f"Chunk_{args.chunk}.zip"
    chunk_path  = extract_dir / f"Chunk_{args.chunk}"

    print(f"Chunk        : {args.chunk}")
    print(f"Base dir     : {base_dir.resolve()}")
    print(f"Zip path     : {zip_path}")
    print(f"Extract path : {chunk_path}")
    print()

    # --- Download ---
    if not args.skip_download:
        zip_path = download_chunk(args.chunk, base_dir, args.hf_token)
    else:
        if not zip_path.exists():
            raise FileNotFoundError(
                f"--skip-download set but zip not found at {zip_path}"
            )
        print(f"Skipping download. Using existing zip: {zip_path}")

    # --- Extract ---
    if not args.skip_extract:
        extract_chunk(zip_path, extract_dir)
    else:
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"--skip-extract set but extracted dir not found at {chunk_path}"
            )
        print(f"Skipping extraction. Using existing dir: {chunk_path}")

    # --- Verify ---
    verify_can_values(chunk_path)
    verify_can_sampling(chunk_path)

    print(f"\nSetup complete for Chunk_{args.chunk}.")


if __name__ == "__main__":
    main()
