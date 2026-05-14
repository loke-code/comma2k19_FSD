# Comma2k19 Data Loader & Visualizer

## Table of Contents
- [Dataset Structure](#dataset-structure)
- [Dependencies](#dependencies)
- [Scripts Overview](#scripts-overview)
- [data\_loader.py — Deep Dive](#data_loaderpy--deep-dive)
  - [Comma\_Segment](#comma_segment)
  - [Comma\_Instance](#comma_instance)
  - [Comma\_Instance\_Temporal](#comma_instance_temporal)
  - [What Each Sample Contains](#what-each-sample-contains)
  - [Usage](#usage)
- [visualizer.py — Deep Dive](#visualizerpy--deep-dive)
  - [Usage](#usage-1)
- [What to Expect at Runtime](#what-to-expect-at-runtime)
- [Known Warnings](#known-warnings)
- [Extending the Pipeline](#extending-the-pipeline)

---

## Dataset Structure

The comma2k19 dataset is organized as follows. Your `chunk_path` should point to a `Chunk_X` directory:

```
Chunk_1/
├── b0c9d2329ad1606b_2018-07-27--06-03-57/     ← drive (dated folder)
│   ├── 1/                                      ← segment (40sec clip)
│   │   ├── video.hevc                          ← raw dashcam footage
│   │   ├── global_pose/
│   │   │   └── frame_times                     ← numpy array, timestamp per frame
│   │   └── processed_log/
│   │       └── CAN/
│   │           ├── steering_angle/
│   │           │   ├── t                       ← numpy array, steering timestamps
│   │           │   └── value                   ← numpy array, steering values (degrees)
│   │           └── speed/
│   │               ├── t                       ← numpy array, speed timestamps
│   │               └── value                   ← numpy array, speed values (m/s)
│   ├── 2/
│   ├── 3/
│   └── ...
├── <another_drive>/
└── ...
```

Each segment is approximately **40 seconds** of driving footage recorded at **~20 fps**, giving roughly **800 frames per segment**.

---

## Dependencies

Install the required packages:

```bash
pip install torch torchvision numpy opencv-python
```

| Package | Purpose |
|---|---|
| `torch` | Tensor operations, Dataset/DataLoader base classes |
| `numpy` | Loading `.npy` CAN signal files, linear interpolation |
| `opencv-python` | Video decoding (HEVC), frame preprocessing, visualization |
| `pathlib` | Clean cross-platform path handling |

---

## Scripts Overview

| Script | Purpose |
|---|---|
| `data_loader.py` | PyTorch-compatible dataset pipeline for model training |
| `visualizer.py` | Standalone viewer to inspect raw footage with overlaid CAN data |

---

## data\_loader.py — Deep Dive

This script defines two classes that work together to produce a stream of training-ready samples from the raw dataset.

---

### Comma\_Segment

```python
segment = Comma_Segment(segment_path, target_size=(256, 256))
```

This class wraps a **single segment folder** and is responsible for:

**1. Loading CAN signals**

The car's CAN bus records speed and steering angle at a high frequency, but these timestamps don't align 1:1 with camera frame timestamps. To solve this, we use `numpy.interp` — given the exact timestamp of a camera frame, it linearly interpolates between the two nearest CAN readings to produce an accurate value at that exact moment.

```
CAN signal (speed):     ──●────────●────────●────────●──
                          t1       t2       t3       t4
Camera frame time:              ▲
                                │
                         interpolated here
```

**2. Preprocessing frames**

Raw frames from the dashcam are `(874, 1164, 3)` — wider than they are tall. Before resizing, we center-crop to a square to preserve the aspect ratio:

```
Original (874 × 1164):
┌──────────────────────────────────────┐
│        │                   │         │
│  crop  │   keep (874×874)  │  crop   │
│  out   │                   │  out    │
└──────────────────────────────────────┘
 ←145px→                       ←145px→

After crop + resize: (256 × 256 × 3)
```

**Key attributes after initialization:**

| Attribute | Type | Description |
|---|---|---|
| `frame_times` | `np.ndarray` | Absolute timestamps for each video frame |
| `steer_t` / `steer_val` | `np.ndarray` | Steering angle timestamps and values (degrees) |
| `speed_t` / `speed_val` | `np.ndarray` | Speed timestamps and values (m/s) |
| `video_path` | `str` | Path to the `.hevc` video file |

---

### Comma\_Instance

```python
dataset = Comma_Instance(chunk_path, target_size=(256, 256), future_time=1.0)
```

This is the main **PyTorch `IterableDataset`** class. It discovers all segments inside a chunk, then streams samples one by one to the DataLoader.

**Constructor arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `chunk_path` | `Path` | required | Path to the `Chunk_X` directory |
| `target_size` | `tuple` | `(256, 256)` | Frame resolution after preprocessing |
| `future_time` | `float` | `1.0` | Seconds ahead to predict CAN values (the label horizon) |

**What `future_time` means:**

Every sample pairs the current frame with CAN readings at `t_current` (input features) and CAN readings at `t_current + future_time` (labels). With the default of `1.0`, the model is trained to predict what the speed and steering angle will be **1 second from now**, given only the current image and current sensor state.

```
              t_current           t_current + 1.0s
                  │                      │
  ────────────────●──────────────────────●────────────
                  ↑                      ↑
              x_speed               y_speed
              x_steer               y_steer
              x_frame
```

Frames near the **end of each segment** where `t + future_time` would exceed the last CAN reading are automatically skipped to prevent extrapolation errors.

---

### What Each Sample Contains

Each item yielded is a dictionary of tensors:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `x_frame` | `(3, 256, 256)` | `float32` | Preprocessed camera frame, normalized to `[0, 1]`, channels-first (C, H, W) |
| `x_speed` | `scalar` | `float32` | Vehicle speed at `t_current` (m/s) |
| `x_steer` | `scalar` | `float32` | Steering angle at `t_current` (degrees) |
| `y_speed` | `scalar` | `float32` | Vehicle speed at `t_current + future_time` (m/s) — **label** |
| `y_steer` | `scalar` | `float32` | Steering angle at `t_current + future_time` (degrees) — **label** |

When batched with `batch_size=32`, shapes become `(32, 3, 256, 256)`, `(32,)`, etc.

---

### Comma_Instance_Temporal
```python
pythondataset = Comma_Instance_Temporal(chunk_path, target_size=(256, 256), future_time=1.0)
```

Extends Comma_Instance to include a short history of CAN signals leading up to the current frame. Everything else — segment discovery, frame loading, memory management — is inherited unchanged.

**Why temporal context?**

A single frame cannot tell you if the car is accelerating, braking, or mid-turn. The CAN history gives the model that context without the cost of stacking multiple frames.

**History samples:**
CAN values are sampled at t-1.5s, t-1.0s, and t-0.5s before the current frame using the same np.interp interpolation used elsewhere. For frames near the start of a segment where history would fall before the first CAN reading, np.interp clamps to the earliest available value automatically.
```
  t-1.5s    t-1.0s    t-0.5s    t_current          t+1.0s
    │          │          │          │                  │
────●──────────●──────────●──────────●──────────────────●────
    ↑          ↑          ↑          ↑                  ↑
   history              history   x_speed            y_speed
                                  x_steer            y_steer
                                  x_frame
```

### What Each Sample Contains

Each item yielded is a dictionary of tensors:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `x_frame` | `(3, 256, 256)` | `float32` | Preprocessed camera frame, normalized to `[0, 1]`, channels-first (C, H, W) |
| `x_speed` | `scalar` | `float32` | Vehicle speed at `t_current` (m/s) |
| `x_steer` | `scalar` | `float32` | Steering angle at `t_current` (degrees) |
| `y_speed` | `scalar` | `float32` | Vehicle speed at `t_current + future_time` (m/s) — **label** |
| `y_steer` | `scalar` | `float32` | Steering angle at `t_current + future_time` (degrees) — **label** |
| `x_speed_history` | `(3,)` | `float32` | Speed at `t-1.5, t-1.0, t-0.5` (m/s) |
| `x_steer_history` | `(3,)` | `float32` | Steer at `t-1.5, t-1.0, t-0.5` (degrees) |

When batched with `batch_size=32`, shapes become `(32, 3, 256, 256)`, `(32,)`, etc.

---

### Usage (Comma_Instance)

**Basic usage:**

```python
from data_utils.data_loader import Comma_Instance
from torch.utils.data import DataLoader
from pathlib import Path

dataset = Comma_Instance(
    chunk_path=Path("comma2k19_data/extracted/Chunk_1"),
    target_size=(256, 256),
    future_time=1.0
)

loader = DataLoader(dataset, batch_size=32, num_workers=0)

for batch in loader:
    frames = batch["x_frame"]   # (32, 3, 256, 256)
    speed  = batch["x_speed"]   # (32,)
    steer  = batch["x_steer"]   # (32,)
    y_spd  = batch["y_speed"]   # (32,)  ← labels
    y_str  = batch["y_steer"]   # (32,)  ← labels
```

**Visualizing a frame from a batch:**

```python
import cv2

frame = batch["x_frame"][0]                     # grab first frame in batch
frame = frame.permute(1, 2, 0).numpy()          # (C,H,W) → (H,W,C)
frame = (frame * 255).astype("uint8")           # [0,1] → [0,255]

cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Using inside a training loop:**

```python
for batch in loader:
    x = batch["x_frame"]                               # image input
    x_can = torch.stack([batch["x_speed"],
                         batch["x_steer"]], dim=1)     # (32, 2) CAN input
    y = torch.stack([batch["y_speed"],
                     batch["y_steer"]], dim=1)         # (32, 2) labels

    predictions = model(x, x_can)
    loss = criterion(predictions, y)
    ...
```

> **Note:** Set `num_workers=0` to start. OpenCV and Python multiprocessing can conflict when `num_workers > 0`, causing hangs or crashes.

---

## visualizer.py — Deep Dive

A standalone diagnostic tool to visually verify that the CAN data is correctly aligned with the video frames. Useful for sanity-checking before training.

**What it renders:**

- Raw dashcam footage (center-cropped to square, full resolution)
- Speed overlay (top-left, in m/s)
- Steering angle overlay (top-left, in degrees)
- A red steering indicator line that rotates in real time with the steering angle — pointing left when turning left, right when turning right
- Press `Q` to quit early.

### Usage

```python
# edit the path at the bottom of visualizer.py
test_path = r"comma2k19_data\extracted\Chunk_1\<drive_folder>\<segment_number>"
```

Then run:

```bash
python data_utils/visualizer.py
```

---

## What to Expect at Runtime

**On first run of `data_loader.py`:**
- Startup takes a few seconds as `_discover_segments()` walks the directory tree and `Comma_Segment` loads all `.npy` files for each segment
- The first batch may be slower as the first video is decoded sequentially into memory
- Subsequent batches within the same segment are fast (already in RAM)

**On `visualizer.py`:**
- A window will open showing the dashcam feed with overlaid telemetry
- Expect the steering indicator to lag slightly on sharp turns — this is normal interpolation behavior at the edges of CAN timestamps

---

## Extending the Pipeline

**Lower resolution for faster training:**
```python
dataset = Comma_Instance(chunk_path, target_size=(128, 128))
```

**Longer prediction horizon:**
```python
dataset = Comma_Instance(chunk_path, future_time=2.0)  # predict 2 seconds ahead
```