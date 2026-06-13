import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
import torch
import random

# ImageNet normalization constants (applied after /255.0)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Prompt thresholds
STEER_THRESHOLD = 1   # radians — below this is considered "straight"
ACCEL_THRESHOLD = 2   # m/s delta vs. recent history to call it accel/decel
ANCHOR_STRIDE = 5

# How far back (in seconds) to sample speed for the acceleration description
SPEED_HISTORY_OFFSETS = [0.5, 1.0, 2.0]
STEER_LO, STEER_HI = -24.28, 21.30 # based on dataset statistics


class Comma_Segment:
    '''
    Processes a single 40-second video segment and its corresponding CAN bus logs.
    '''
    def __init__(self, segment_path: Path, target_size=(224, 224)):
        self.segment_path = segment_path
        self.target_size = target_size

        self.frame_times = np.load(segment_path / "global_pose" / "frame_times").flatten()
        self.steer_t     = np.load(segment_path / "processed_log" / "CAN" / "steering_angle" / "t").flatten()
        self.steer_val   = np.load(segment_path / "processed_log" / "CAN" / "steering_angle" / "value").flatten()
        self.speed_t     = np.load(segment_path / "processed_log" / "CAN" / "speed" / "t").flatten()
        self.speed_val   = np.load(segment_path / "processed_log" / "CAN" / "speed" / "value").flatten()

        self.video_path = str(segment_path / "video.hevc")
        self.can_t_min = max(self.speed_t[0], self.steer_t[0])
        self.can_t_max = min(self.speed_t[-1], self.steer_t[-1])

    def get_CAN_data(self, t):
        ''' Interpolates speed (m/s) and steer (rad) at exact timestamp t '''
        speed = np.interp(t, self.speed_t, self.speed_val)
        steer = np.interp(t, self.steer_t, self.steer_val)
        return float(speed), float(steer)

    def preprocess_frame(self, frame):
        ''' Center-crops to 1:1 aspect ratio, resizes, converts BGR->RGB, normalizes to [0,1] '''
        h, w, _ = frame.shape
        start_x = (w - h) // 2
        frame = frame[:, start_x : start_x + h]
        frame = cv2.resize(frame, self.target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.frame_times)


class Comma_Continuous_VLA(IterableDataset):
    '''
    Outputs temporal visual history stacks (chronological order), explicit low-dimensional
    state vectors, a continuous action trajectory tensor, and a dynamic language prompt
    describing the vehicle's current steering and speed behavior.
    '''
    def __init__(self, chunk_path: Path, target_size=(224, 224),
                 num_history_frames=4, frame_stride=5,
                 future_window=1.0, num_future_steps=5, split="train"): # <-- Added split arg
        
        self.chunk_path = Path(chunk_path)
        self.target_size = target_size
        self.num_history_frames = num_history_frames
        self.frame_stride = frame_stride
        self.future_window = future_window
        self.num_future_steps = num_future_steps
        self.split = split # "train", "test", or "all"

        self.future_offsets = np.linspace(
            future_window / num_future_steps, future_window, num_future_steps
        )

        self.segment_paths = self._discover_segments()

    def _discover_segments(self):
        segments = []
        for drive in sorted(self.chunk_path.iterdir()):
            if not drive.is_dir():
                continue
            for seg_path in sorted(
                drive.iterdir(),
                key=lambda x: int(x.name) if x.name.isdigit() else x.name
            ):
                if seg_path.is_dir():
                    segments.append(seg_path)

        # --- INTERNAL SPLIT LOGIC ---
        total_segments = len(segments)
        split_idx = int(0.9 * total_segments)

        if self.split == "train":
            assigned_segments = segments[:split_idx]
        elif self.split == "test":
            assigned_segments = segments[split_idx:]
        else:
            assigned_segments = segments # Default to all if split="all"

        print(f"[Dataset] Initialized '{self.split}' split with {len(assigned_segments)} segments.")
        return assigned_segments

    def _load_segment_frames(self, segment):
        cap = cv2.VideoCapture(segment.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(segment.preprocess_frame(frame))
        cap.release()
        return frames  # List of (H, W, 3) float32 arrays, already in [0, 1]

    def _build_prompt(self, x_speed, x_steer, t_current, segment):
        """
        Builds a context-aware prompt describing what the vehicle is currently doing.
        Steering direction is inferred from sign and magnitude of steer angle.
        Speed behavior (accel / decel / cruise) is inferred by comparing current speed
        against recent CAN history — only using timestamps safely inside the CAN window.
        """
        if abs(x_steer) < STEER_THRESHOLD:
            steer_desc = "going straight"
        elif x_steer > 0:
            steer_desc = f"steering right ({x_steer:.2f} degrees)"
        else:
            steer_desc = f"steering left ({x_steer:.2f} degrees)"

        past_speeds = []
        for offset in SPEED_HISTORY_OFFSETS:
            t_past = t_current - offset
            if t_past >= segment.can_t_min:
                v_past, _ = segment.get_CAN_data(t_past)
                past_speeds.append(v_past)

        if past_speeds:
            avg_past_speed = float(np.mean(past_speeds))
            delta = x_speed - avg_past_speed
            if delta > ACCEL_THRESHOLD:
                speed_desc = f"accelerating (current {x_speed:.2f} m/s, avg recent {avg_past_speed:.2f} m/s)"
            elif delta < -ACCEL_THRESHOLD:
                speed_desc = f"decelerating (current {x_speed:.2f} m/s, avg recent {avg_past_speed:.2f} m/s)"
            else:
                speed_desc = f"maintaining speed at {x_speed:.2f} m/s"
        else:
            speed_desc = f"traveling at {x_speed:.2f} m/s"

        return (
            f"The vehicle is {steer_desc} and {speed_desc}. "
            f"Predict the driving trajectory for the next {self.future_window:.1f} seconds."
        )

    # Main iteration

    def __iter__(self):
        shuffled_segments = self.segment_paths.copy()
        random.shuffle(shuffled_segments)

        for seg_path in shuffled_segments:
            segment = Comma_Segment(seg_path, self.target_size)

            frames = self._load_segment_frames(segment)
            total_frames = len(frames)

            min_valid_idx = (self.num_history_frames - 1) * self.frame_stride
            if min_valid_idx >= total_frames:
                continue

            frame_indices = list(range(min_valid_idx, total_frames, ANCHOR_STRIDE))
            random.shuffle(frame_indices)

            for frame_idx in frame_indices:
                t_current = segment.frame_times[frame_idx]

                # frame timestamp must sit inside the CAN log on both ends
                if t_current < segment.can_t_min:
                    continue
                if t_current + self.future_window > segment.can_t_max:
                    continue
                if t_current - SPEED_HISTORY_OFFSETS[-1] < segment.can_t_min:
                    continue

                #    i counts down from (num_history_frames-1) to 0, so:
                #    history_frames[0] = oldest,  history_frames[-1] = current
                history_frames = []
                for i in range(self.num_history_frames - 1, -1, -1):
                    past_idx = frame_idx - (i * self.frame_stride)
                    frame_tensor = torch.from_numpy(frames[past_idx]).permute(2, 0, 1)
                    # ImageNet normalization
                    frame_tensor = (frame_tensor - IMAGENET_MEAN) / IMAGENET_STD
                    history_frames.append(frame_tensor)

                # Shape: (num_history_frames, 3, H, W)
                pixel_values_stack = torch.stack(history_frames, dim=0)

                x_speed, x_steer = segment.get_CAN_data(t_current)
                x_speed = np.clip(x_speed, 0.0, 40.0)
                x_steer = np.clip(x_steer, STEER_LO, STEER_HI)
                state_history = [[x_speed, x_steer]]  # current frame first
                for i in range(1, self.num_history_frames):
                    t_past = segment.frame_times[frame_idx - (i * self.frame_stride)]
                    v_past, s_past = segment.get_CAN_data(t_past)
                    s_past = np.clip(s_past, STEER_LO, STEER_HI)
                    v_past = np.clip(v_past, 0.0, 40.0)
                    state_history.append([v_past, s_past])
                state_history.reverse()  # reorder oldest -> newest, matching visual stack
                observation_state = torch.tensor(state_history, dtype=torch.float32)  # (num_history_frames, 2)

                trajectory_actions = []
                for offset in self.future_offsets:
                    v_f, alpha_f = segment.get_CAN_data(t_current + offset)
                    trajectory_actions.append([v_f, alpha_f])

                trajectory_actions = [[np.clip(v, 0.0, 40.0), np.clip(a, STEER_LO, STEER_HI)] for v, a in trajectory_actions]

                # Shape: (num_future_steps, 2)
                actions_tensor = torch.tensor(trajectory_actions, dtype=torch.float32)

                prompt_str = self._build_prompt(x_speed, x_steer, t_current, segment)

                yield {
                    "pixel_values": pixel_values_stack,   # (N_frames, 3, H, W)
                    "observation": {
                        "state": observation_state,        
                    },
                    "actions": actions_tensor,             # (N_future_steps, 2)
                    "prompt": prompt_str,
                }

            del frames


if __name__ == "__main__":
    chunk = Path("comma2k19_data") / "extracted" / "Chunk_1"

    dataset = Comma_Continuous_VLA(
        chunk,
        target_size=(224, 224),
        num_history_frames=5,
        frame_stride=5,
        future_window=1.0,
        num_future_steps=5
    )

    loader = DataLoader(dataset, batch_size=2, num_workers=0)

    for batch in loader:
        print("--- Continuous Tensor VLA Data Stream Check ---")
        print(f"Visual History Stack Shape:   {batch['pixel_values'].shape}")   # (B, N_frames, 3, H, W)
        print(f"Observation State Tensor:     {batch['observation']['state']}")  # (B, 2)
        print(f"Continuous Action Trajectory: {batch['actions'].shape}")          # (B, N_future_steps, 2)
        print(f"Sample Prompt:\n  {batch['prompt'][0]}")
        print(batch.keys())
        break