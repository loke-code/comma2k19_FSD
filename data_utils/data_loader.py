from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


def _to_float_tensor(value: float | list[float]) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def _rgb_frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


def _pad_or_truncate_state(state_values: list[float], target_dim: int | None) -> torch.Tensor:
    state = _to_float_tensor(state_values)
    if target_dim is None:
        return state
    if state.numel() < target_dim:
        state = F.pad(state, (0, target_dim - state.numel()))
    elif state.numel() > target_dim:
        state = state[:target_dim]
    return state


class Comma_Segment:
    """
    Process one comma2k19 segment: video frames plus aligned CAN signals.

    Shapes:
    - raw decoded frame from OpenCV: (H_raw, W_raw, 3), uint8, BGR
    - preprocessed frame returned by preprocess_frame: (H, W, 3), uint8, RGB
      where (W, H) == target_size
    """

    def __init__(self, segment_path: Path, target_size: tuple[int, int] = (256, 256)):
        self.segment_path = Path(segment_path)
        self.target_size = target_size

        self.frame_times = np.load(segment_path / "global_pose" / "frame_times").flatten()
        self.steer_t = np.load(segment_path / "processed_log" / "CAN" / "steering_angle" / "t").flatten()
        self.steer_val = np.load(segment_path / "processed_log" / "CAN" / "steering_angle" / "value").flatten()
        self.speed_t = np.load(segment_path / "processed_log" / "CAN" / "speed" / "t").flatten()
        self.speed_val = np.load(segment_path / "processed_log" / "CAN" / "speed" / "value").flatten()
        self.video_path = str(segment_path / "video.hevc")

    def get_CAN_data(self, timestamp: float) -> tuple[float, float]:
        speed = np.interp(timestamp, self.speed_t, self.speed_val)
        steer = np.interp(timestamp, self.steer_t, self.steer_val)
        return float(speed), float(steer)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Center-crop the wide frame to a square, resize, and convert BGR -> RGB.
        """
        height, width, _ = frame.shape
        start_x = (width - height) // 2
        frame = frame[:, start_x : start_x + height]
        frame = cv2.resize(frame, self.target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __len__(self) -> int:
        return len(self.frame_times)


class Comma_Instance(IterableDataset):
    """
    Single-step dataset for PyTorch, inference, and LeRobot-style preprocessing.

    Each yielded sample represents one timestep with:
    - input = current frame and current / past vehicle state
    - target = future driving action after future_time seconds

    Shapes per sample:
    - x_frame: (3, H, W) input image
    - x_state: (D,) input state
    - y_action: (2,) target action
    - observation["image"]: (3, H, W)
    - observation["state"]: (D,)
    - action: (2,)
    - x_speed, x_steer: input scalar tensors with shape ()
    - y_speed, y_steer: target scalar tensors with shape ()
    - x_speed_history, x_steer_history: input history tensors with shape (K) when history is enabled

    Definitions:
    - H, W come from target_size
    - K = len(history_offsets)
    - D = 2 + 2K

    With a PyTorch DataLoader(batch_size=B), these become:
    - x_frame: (B, 3, H, W)
    - x_state: (B, D)
    - y_action: (B, 2)
    """

    def __init__(
        self,
        chunk_path: Path,
        target_size: tuple[int, int] = (256, 256),
        future_time: float = 1.0,
        history_offsets: list[float] | None = None,
        policy_visual_keys: list[str] | None = None,
        policy_state_key: str | None = None,
        policy_state_dim: int | None = None,
        task: str = "",
        robot_type: str = "comma2k19_car",
        shuffle_segments: bool = True,
        shuffle_frames: bool = True,
    ):
        self.chunk_path = Path(chunk_path)
        self.target_size = target_size
        self.future_time = future_time
        self.history_offsets = history_offsets or []
        self.policy_visual_keys = list(policy_visual_keys or [])
        self.policy_state_key = policy_state_key
        self.policy_state_dim = policy_state_dim
        self.task = task
        self.robot_type = robot_type
        self.shuffle_segments = shuffle_segments
        self.shuffle_frames = shuffle_frames
        self.segment_paths = self._discover_segments()

    def _discover_segments(self) -> list[Path]:
        segments: list[Path] = []
        for drive in sorted(self.chunk_path.iterdir()):
            if not drive.is_dir():
                continue
            for seg_path in sorted(drive.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else x.name):
                if seg_path.is_dir():
                    segments.append(seg_path)
        return segments

    def _ordered_segments(self) -> list[Path]:
        segments = self.segment_paths.copy()
        if self.shuffle_segments:
            random.shuffle(segments)
        return segments

    def _ordered_frame_indices(self, num_frames: int) -> list[int]:
        indices = list(range(num_frames))
        if self.shuffle_frames:
            random.shuffle(indices)
        return indices

    def _load_segment_frames(self, segment: Comma_Segment) -> list[np.ndarray]:
        cap = cv2.VideoCapture(segment.video_path)
        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(segment.preprocess_frame(frame))
        cap.release()
        return frames

    def build_state_vector(
        self,
        x_speed: float,
        x_steer: float,
        speed_history: list[float],
        steer_history: list[float],
        target_dim: int | None = None,
    ) -> torch.Tensor:
        state_values = [x_speed, x_steer]
        for speed_value, steer_value in zip(speed_history, steer_history):
            state_values.extend([speed_value, steer_value])
        return _pad_or_truncate_state(state_values, target_dim)

    def build_policy_observation(
        self,
        *,
        image: torch.Tensor,
        state: torch.Tensor,
        timestamp: float,
    ) -> dict[str, Any] | None:
        if not self.policy_visual_keys:
            return None

        observation: dict[str, Any] = {}
        for key in self.policy_visual_keys:
            observation[key] = image.clone()

        if self.policy_state_key is not None:
            target_state = state
            if self.policy_state_dim is not None:
                target_state = _pad_or_truncate_state(state.tolist(), self.policy_state_dim)
            observation[self.policy_state_key] = target_state

        observation["task"] = self.task
        observation["robot_type"] = self.robot_type
        observation["timestamp"] = float(timestamp)
        return observation

    def build_sample(
        self,
        segment: Comma_Segment,
        frame: np.ndarray,
        frame_idx: int,
    ) -> dict[str, Any] | None:
        t_current = float(segment.frame_times[frame_idx])
        t_future = t_current + self.future_time
        max_can_time = min(segment.speed_t[-1], segment.steer_t[-1])
        if t_future > max_can_time:
            return None

        x_speed, x_steer = segment.get_CAN_data(t_current)
        y_speed, y_steer = segment.get_CAN_data(t_future)
        speed_history = [segment.get_CAN_data(t_current - offset)[0] for offset in self.history_offsets]
        steer_history = [segment.get_CAN_data(t_current - offset)[1] for offset in self.history_offsets]

        image = _rgb_frame_to_tensor(frame)
        state = self.build_state_vector(x_speed, x_steer, speed_history, steer_history)
        action = _to_float_tensor([y_speed, y_steer])

        sample: dict[str, Any] = {
            "x_frame": image,
            "x_state": state,
            "y_action": action,
            "observation": {
                "image": image,
                "state": state,
            },
            "action": action,
            "x_speed": _to_float_tensor(x_speed),
            "x_steer": _to_float_tensor(x_steer),
            "y_speed": _to_float_tensor(y_speed),
            "y_steer": _to_float_tensor(y_steer),
            "timestamp": t_current,
        }

        if speed_history:
            sample["x_speed_history"] = _to_float_tensor(speed_history)
            sample["x_steer_history"] = _to_float_tensor(steer_history)

        policy_observation = self.build_policy_observation(image=image, state=state, timestamp=t_current)
        if policy_observation is not None:
            sample["policy_observation"] = policy_observation

        return sample

    def iter_segment_samples(self, segment_path: Path) -> list[dict[str, Any]]:
        segment = Comma_Segment(segment_path, self.target_size)
        frames = self._load_segment_frames(segment)
        samples: list[dict[str, Any]] = []
        for frame_idx in self._ordered_frame_indices(len(frames)):
            sample = self.build_sample(segment, frames[frame_idx], frame_idx)
            if sample is not None:
                samples.append(sample)
        del frames
        return samples

    def __iter__(self):
        for seg_path in self._ordered_segments():
            for sample in self.iter_segment_samples(seg_path):
                yield sample


class Comma_CAN_Temporal(Comma_Instance):
    """
    Single-step dataset with CAN history folded into the state vector.

    Default history offsets are [0.1, 0.4, 1.0, 2.0], so by default:
    - K = 4
    - x_state / observation["state"]: (10,)
    - x_speed_history: (4,)
    - x_steer_history: (4,)
    """

    def __init__(
        self,
        chunk_path: Path,
        target_size: tuple[int, int] = (256, 256),
        future_time: float = 1.0,
        history_offsets: list[float] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            chunk_path=chunk_path,
            target_size=target_size,
            future_time=future_time,
            history_offsets=history_offsets or [0.1, 0.4, 1.0, 2.0],
            **kwargs,
        )


class LeRobotTrajectory:
    """
    One full segment represented as a trajectory of preprocessed samples.

    Per-step shapes from get_step():
    - observation["image"]: (3, H, W)
    - observation["state"]: (D,)
    - action: (2,)

    Full-episode shapes from to_episode_dict():
    - observations["image"]: (T, 3, H, W)
    - observations["state"]: (T, D)
    - actions: (T, 2)

    Definitions:
    - T = number of valid timesteps in the segment
    - D = 2 + 2K
    - K = len(history_offsets)
    """

    def __init__(
        self,
        segment: Comma_Segment,
        frames: list[np.ndarray],
        future_time: float = 1.0,
        history_offsets: list[float] | None = None,
        policy_visual_keys: list[str] | None = None,
        policy_state_key: str | None = None,
        policy_state_dim: int | None = None,
        task: str = "",
        robot_type: str = "comma2k19_car",
    ):
        self.segment = segment
        self.frames = frames
        self.future_time = future_time
        self.history_offsets = history_offsets or []
        self.policy_visual_keys = list(policy_visual_keys or [])
        self.policy_state_key = policy_state_key
        self.policy_state_dim = policy_state_dim
        self.task = task
        self.robot_type = robot_type
        self.frame_times = segment.frame_times
        self.max_can_time = min(segment.speed_t[-1], segment.steer_t[-1])
        self.valid_indices = [
            i for i in range(len(frames))
            if self.frame_times[i] + future_time <= self.max_can_time
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _build_state_vector(
        self,
        x_speed: float,
        x_steer: float,
        speed_history: list[float],
        steer_history: list[float],
        target_dim: int | None = None,
    ) -> torch.Tensor:
        state_values = [x_speed, x_steer]
        for speed_value, steer_value in zip(speed_history, steer_history):
            state_values.extend([speed_value, steer_value])
        return _pad_or_truncate_state(state_values, target_dim)

    def _build_policy_observation(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        timestamp: float,
    ) -> dict[str, Any] | None:
        if not self.policy_visual_keys:
            return None

        observation: dict[str, Any] = {}
        for key in self.policy_visual_keys:
            observation[key] = image.clone()

        if self.policy_state_key is not None:
            target_state = state
            if self.policy_state_dim is not None:
                target_state = _pad_or_truncate_state(state.tolist(), self.policy_state_dim)
            observation[self.policy_state_key] = target_state

        observation["task"] = self.task
        observation["robot_type"] = self.robot_type
        observation["timestamp"] = float(timestamp)
        return observation

    def get_sample(self, idx: int) -> dict[str, Any]:
        frame_idx = self.valid_indices[idx]
        timestamp = float(self.frame_times[frame_idx])
        future_timestamp = timestamp + self.future_time

        x_speed, x_steer = self.segment.get_CAN_data(timestamp)
        y_speed, y_steer = self.segment.get_CAN_data(future_timestamp)
        speed_history = [self.segment.get_CAN_data(timestamp - offset)[0] for offset in self.history_offsets]
        steer_history = [self.segment.get_CAN_data(timestamp - offset)[1] for offset in self.history_offsets]

        image = _rgb_frame_to_tensor(self.frames[frame_idx])
        state = self._build_state_vector(x_speed, x_steer, speed_history, steer_history)
        action = _to_float_tensor([y_speed, y_steer])

        sample: dict[str, Any] = {
            "observation": {
                "image": image,
                "state": state,
            },
            "action": action,
            "timestamp": timestamp,
        }

        if speed_history:
            sample["x_speed_history"] = _to_float_tensor(speed_history)
            sample["x_steer_history"] = _to_float_tensor(steer_history)

        policy_observation = self._build_policy_observation(image, state, timestamp)
        if policy_observation is not None:
            sample["policy_observation"] = policy_observation

        return sample

    def get_step(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        sample = self.get_sample(idx)
        return sample["observation"], sample["action"]

    def to_episode_dict(self) -> dict[str, Any]:
        images: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        policy_observations: list[dict[str, Any]] = []

        for idx in range(len(self)):
            sample = self.get_sample(idx)
            images.append(sample["observation"]["image"])
            states.append(sample["observation"]["state"])
            actions.append(sample["action"])
            if "policy_observation" in sample:
                policy_observations.append(sample["policy_observation"])

        episode = {
            "observations": {
                "image": torch.stack(images),
                "state": torch.stack(states),
            },
            "actions": torch.stack(actions),
            "length": len(self),
        }

        if policy_observations:
            episode["policy_observations"] = policy_observations

        return episode


class CommaLeRobotTrajectories(IterableDataset):
    """
    Yield one full preprocessed episode per segment.

    Each yielded item has shapes:
    - observations["image"]: (T, 3, H, W)
    - observations["state"]: (T, D)
    - actions: (T, 2)
    - length: int

    If policy_visual_keys are configured, policy_observations is also
    included as a Python list of length T, where each entry is a timestep
    dictionary containing per-camera tensors of shape (3, H, W) and an
    optional policy state tensor of shape (P,).
    """

    def __init__(
        self,
        chunk_path: Path,
        target_size: tuple[int, int] = (256, 256),
        future_time: float = 1.0,
        history_offsets: list[float] | None = None,
        policy_visual_keys: list[str] | None = None,
        policy_state_key: str | None = None,
        policy_state_dim: int | None = None,
        task: str = "",
        robot_type: str = "comma2k19_car",
        shuffle_segments: bool = True,
    ):
        self.base = Comma_Instance(
            chunk_path=chunk_path,
            target_size=target_size,
            future_time=future_time,
            history_offsets=history_offsets,
            policy_visual_keys=policy_visual_keys,
            policy_state_key=policy_state_key,
            policy_state_dim=policy_state_dim,
            task=task,
            robot_type=robot_type,
            shuffle_segments=shuffle_segments,
            shuffle_frames=False,
        )
        self.target_size = target_size
        self.future_time = future_time
        self.history_offsets = history_offsets or []
        self.policy_visual_keys = list(policy_visual_keys or [])
        self.policy_state_key = policy_state_key
        self.policy_state_dim = policy_state_dim
        self.task = task
        self.robot_type = robot_type
        self.segment_paths = self.base.segment_paths
        self.shuffle_segments = shuffle_segments

    def __iter__(self):
        for seg_path in self.base._ordered_segments():
            segment = Comma_Segment(seg_path, self.target_size)
            frames = self.base._load_segment_frames(segment)
            trajectory = LeRobotTrajectory(
                segment=segment,
                frames=frames,
                future_time=self.future_time,
                history_offsets=self.history_offsets,
                policy_visual_keys=self.policy_visual_keys,
                policy_state_key=self.policy_state_key,
                policy_state_dim=self.policy_state_dim,
                task=self.task,
                robot_type=self.robot_type,
            )
            yield trajectory.to_episode_dict()
            del frames


class CommaLeRobotDataset(IterableDataset):
    """
    Unified dataset:
    - mode="frame" yields one timestep at a time for inference or framewise training.
    - mode="episode" yields one full trajectory at a time for sequence training.

    Output shapes depend on mode.

    In mode="frame":
    - x_frame: (3, H, W) input image
    - x_state: (D,) input state
    - y_action: (2,) target action
    - observation["image"]: (3, H, W)
    - observation["state"]: (D,)
    - action: (2,)

    In mode="episode":
    - observations["image"]: (T, 3, H, W)
    - observations["state"]: (T, D)
    - actions: (T, 2)

    Definitions:
    - H, W come from target_size
    - K = len(history_offsets)
    - D = 2 + 2K
    - T = number of valid timesteps in a segment

    When policy mapping is enabled:
    - each policy visual key holds (3, H, W) in frame mode
    - the policy state key holds (P,), where P = policy_state_dim
    - under a PyTorch DataLoader(batch_size=B), these batch to (B, 3, H, W)
      and (B, P) respectively
    """

    def __init__(
        self,
        chunk_path: Path,
        target_size: tuple[int, int] = (256, 256),
        future_time: float = 1.0,
        history_offsets: list[float] | None = None,
        policy_visual_keys: list[str] | None = None,
        policy_state_key: str | None = None,
        policy_state_dim: int | None = None,
        task: str = "",
        robot_type: str = "comma2k19_car",
        mode: str = "frame",
        shuffle_segments: bool = True,
        shuffle_frames: bool = True,
    ):
        if mode not in {"frame", "episode"}:
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'frame' or 'episode'.")

        self.mode = mode
        self.frame_dataset = Comma_Instance(
            chunk_path=chunk_path,
            target_size=target_size,
            future_time=future_time,
            history_offsets=history_offsets,
            policy_visual_keys=policy_visual_keys,
            policy_state_key=policy_state_key,
            policy_state_dim=policy_state_dim,
            task=task,
            robot_type=robot_type,
            shuffle_segments=shuffle_segments,
            shuffle_frames=shuffle_frames,
        )
        self.episode_dataset = CommaLeRobotTrajectories(
            chunk_path=chunk_path,
            target_size=target_size,
            future_time=future_time,
            history_offsets=history_offsets,
            policy_visual_keys=policy_visual_keys,
            policy_state_key=policy_state_key,
            policy_state_dim=policy_state_dim,
            task=task,
            robot_type=robot_type,
            shuffle_segments=shuffle_segments,
        )

    def __iter__(self):
        if self.mode == "frame":
            yield from self.frame_dataset
            return
        yield from self.episode_dataset


if __name__ == "__main__":
    chunk = Path("comma2k19_data/extracted/Chunk_1")
    dataset = CommaLeRobotDataset(chunk, target_size=(256, 256), future_time=1.0, mode="frame")
    loader = DataLoader(dataset, batch_size=4, num_workers=0)

    for batch in loader:
        print(batch["x_frame"].shape)
        print(batch["x_state"].shape)
        print(batch["y_action"].shape)
        break
