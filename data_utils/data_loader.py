import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
import torch
import random


class Comma_Segment:
    '''
    This class is for processing each segment (the subfolders named 1,2,4 etc within a dated folder inside chunk).
    Each segment corresponds to a single 40sec video, and the corresponding sensor readings.
    '''
    def __init__(self, segment_path: Path, target_size=(256, 256)):
        self.segment_path = segment_path
        self.target_size = target_size # we can lower resultion of images for faster training

        # frame_times is the exact time light hits the lens of the camera
        # we will use this to map the respective camera frames to sensor readings 
        self.frame_times = np.load(segment_path / "global_pose" / "frame_times").flatten()
        # steer_t is the time, and steer_val is the steering value at that time
        self.steer_t     = np.load(segment_path / "processed_log" / "CAN" / "steering_angle" / "t").flatten()
        self.steer_val   = np.load(segment_path / "processed_log" / "CAN" / "steering_angle" / "value").flatten()
        # speed_t is the time, and speed_val is the speed value at that time
        self.speed_t     = np.load(segment_path / "processed_log" / "CAN" / "speed" / "t").flatten()
        self.speed_val   = np.load(segment_path / "processed_log" / "CAN" / "speed" / "value").flatten()

        # video is saved in hevc format
        self.video_path = str(segment_path / "video.hevc")

    def get_CAN_data(self, t):
        '''
        t -> frame_times, we take closest speed_t to this frame_times, and interpolate 
        the missing part of speed_val to match the exact frame_times
        '''
        speed = np.interp(t, self.speed_t, self.speed_val)
        steer = np.interp(t, self.steer_t, self.steer_val)
        return float(speed), float(steer)

    def preprocess_frame(self, frame):
        '''
        original image resolution is (874, 1164, 3)
        we convert this to square (1:1 aspect ratio) for easy downsizing (reducing resolution) later
        to do this, we center crop the image frame (cut out the excess width from the sides, while keeping 
        the same height=874)
        '''
        h, w, _ = frame.shape
        start_x = (w - h) // 2
        frame = frame[:, start_x : start_x + h]
        frame = cv2.resize(frame, self.target_size)
        return frame

    def __len__(self):
        return len(self.frame_times)

class Comma_Instance(IterableDataset):
    '''
    This class will provide a single 'Instance' to train
    single instance -> 1 image frame, corresponding speed and steer & the speed and steer the next second
    so there is no temporal information (previous data)
    '''
    def __init__(self, chunk_path: Path, target_size=(256, 256), future_time=1.0):
        self.chunk_path = Path(chunk_path)
        self.target_size = target_size
        self.future_time = future_time
        self.segment_paths = self._discover_segments()

    def _discover_segments(self):
        '''
        This method is to collect and store all the segment paths
        '''
        segments = []
        for drive in sorted(self.chunk_path.iterdir()):
            if not drive.is_dir():
                continue
            for seg_path in sorted(drive.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else x.name):
                if seg_path.is_dir():
                    segments.append(seg_path)
        return segments

    def _load_segment_frames(self, segment):
        """Read all frames from a segment"""
        cap = cv2.VideoCapture(segment.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # append processed frame
            frames.append(segment.preprocess_frame(frame))
        cap.release()
        return frames  # list of (256, 256, 3) uint8 arrays

    def __iter__(self):
        ''' Logic of how we iterate though each segment '''
        # loops through all segments in given dated subfolder within chunk folder

        # shuffle the order we visit segments
        shuffled_segments = self.segment_paths.copy()
        random.shuffle(shuffled_segments)

        for seg_path in shuffled_segments:
            # segment object to process this particular segment
            segment = Comma_Segment(seg_path, self.target_size)
            # to avoid indexing or overflow errors, stop at max
            max_can_time = min(segment.speed_t[-1], segment.steer_t[-1])

            frames = self._load_segment_frames(segment)  # load whole segment at once

            # shuffle frames, 
            frame_indices = list(range(len(frames)))
            random.shuffle(frame_indices)

            # iterate through each frame in the video
            for frame_idx in frame_indices:
                frame = frames[frame_idx]
                # current frame recorded time (t), and predicting time (t+1)
                t_current = segment.frame_times[frame_idx]
                t_plus_1 = t_current + self.future_time

                # to avoid overflow
                if t_plus_1 > max_can_time:
                    continue
                
                # pull the CAN data
                x_speed, x_steer = segment.get_CAN_data(t_current)
                y_speed,  y_steer  = segment.get_CAN_data(t_plus_1)

                '''
                we have processed the entire segment, we pass this out as our dataset
                before moving to the next segment, using yield, we wait until more data is requested
                this way, at any time, we only have one segment loaded in memory at any time
                '''
                yield {
                    "x_frame": torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0,
                    "x_speed": torch.tensor(x_speed, dtype=torch.float32),
                    "x_steer": torch.tensor(x_steer, dtype=torch.float32),
                    "y_speed": torch.tensor(y_speed,  dtype=torch.float32),
                    "y_steer": torch.tensor(y_steer,  dtype=torch.float32),
                }
            
            del frames  # free memory

class Comma_CAN_Temporal(Comma_Instance):
    '''
    Extends Comma_Instance to include CAN history.
    History samples at t-0.5, t-1.0, t-1.5 seconds before current frame.
    '''
    def __init__(self, chunk_path: Path, target_size=(256, 256), future_time=1.0):
        super().__init__(chunk_path, target_size, future_time)
        self.history_offsets = [0.1, 0.4, 1.0, 2.0]  # seconds in past, ordered newest → oldest

    def __iter__(self):
        # shuffle the order we visit segments
        shuffled_segments = self.segment_paths.copy()
        random.shuffle(shuffled_segments)

        for seg_path in shuffled_segments:
            segment = Comma_Segment(seg_path, self.target_size)
            max_can_time = min(segment.speed_t[-1], segment.steer_t[-1])

            frames = self._load_segment_frames(segment)

            # shuffle the frames ordering aswell using indices
            frame_indices = list(range(len(frames)))
            random.shuffle(frame_indices)

            for frame_idx in frame_indices:

                frame = frames[frame_idx]
                t_current = segment.frame_times[frame_idx]
                t_future  = t_current + self.future_time

                # skip if t+1 doesn't exist (we are less than 1 second before the video ends)
                if t_future > max_can_time:
                    continue
                
                # current speed and steer
                x_speed, x_steer = segment.get_CAN_data(t_current)
                y_speed, y_steer = segment.get_CAN_data(t_future)

                # history: sample CAN at t-0.1, t-0.4, t-1, t-2
                # np.interp clamps to boundary if t_past < first CAN timestamp, so early frames are safe
                speed_history = [segment.get_CAN_data(t_current - offset)[0] for offset in self.history_offsets]
                steer_history = [segment.get_CAN_data(t_current - offset)[1] for offset in self.history_offsets]

                yield {
                    "x_frame":         torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0,
                    "x_speed":         torch.tensor(x_speed,       dtype=torch.float32),
                    "x_steer":         torch.tensor(x_steer,       dtype=torch.float32),
                    "x_speed_history": torch.tensor(speed_history, dtype=torch.float32),  # (4,)
                    "x_steer_history": torch.tensor(steer_history, dtype=torch.float32),  # (4,)
                    "y_speed":         torch.tensor(y_speed,       dtype=torch.float32),
                    "y_steer":         torch.tensor(y_steer,       dtype=torch.float32),
                }

            del frames


class LeRobotTrajectory:
    """
    Represents one full trajectory (segment) in LeRobot format.
    """

    def __init__(self, segment: Comma_Segment, frames, future_time=1.0):
        self.segment = segment
        self.frames = frames
        self.future_time = future_time

        self.frame_times = segment.frame_times
        self.max_can_time = min(segment.speed_t[-1], segment.steer_t[-1])

        # precompute valid indices (temporal consistency preserved)
        self.valid_indices = [
            i for i in range(len(frames))
            if self.frame_times[i] + future_time <= self.max_can_time
        ]

    def __len__(self):
        return len(self.valid_indices)

    def get_step(self, idx):
        """
        Returns one LeRobot-style timestep:
        observation + action
        """
        frame_idx = self.valid_indices[idx]

        t = self.frame_times[frame_idx]
        t_future = t + self.future_time

        frame = self.frames[frame_idx]

        x_speed, x_steer = self.segment.get_CAN_data(t)
        y_speed, y_steer = self.segment.get_CAN_data(t_future)

        obs = {
            "image": torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0, # OpenCV is BGR, not RGB
            "state": torch.tensor([x_speed, x_steer], dtype=torch.float32),
        }

        action = torch.tensor([y_speed, y_steer], dtype=torch.float32)

        return obs, action

class CommaLeRobotTrajectories(IterableDataset):
    """
    Returns full trajectories (episodes), not single frames.
    """

    def __init__(self, chunk_path, target_size=(256, 256), future_time=1.0):
        self.base = Comma_Instance(chunk_path, target_size, future_time)
        self.target_size = target_size
        self.future_time = future_time
        self.segment_paths = self.base.segment_paths

    def __iter__(self):
        segments = self.segment_paths.copy()
        random.shuffle(segments)

        for seg_path in segments:
            segment = Comma_Segment(seg_path, self.target_size)
            frames = self.base._load_segment_frames(segment)

            traj = LeRobotTrajectory(
                segment=segment,
                frames=frames,
                future_time=self.future_time
            )

            observations = []
            actions = []

            for i in range(len(traj)):
                obs, act = traj.get_step(i)
                observations.append(obs)
                actions.append(act)

            yield {
                "observations": observations,
                "actions": torch.stack(actions),
                "length": len(traj),
            }



    class CommaLeRobotDataset(IterableDataset):
        """
        Single PyTorch + LeRobot-compatible dataset.
    
        Uses:
            Comma_Segment
            LeRobotTrajectory
    
        Returns:
            observations: (T, 3, H, W), (T, 2)
            actions: (T, 2)
        """
    
        def __init__(self, chunk_path, target_size=(256, 256), future_time=1.0):
            self.base = Comma_Instance(chunk_path, target_size, future_time)
            self.segment_paths = self.base.segment_paths
            self.target_size = target_size
            self.future_time = future_time
    
        def __iter__(self):
            segments = self.segment_paths.copy()
            random.shuffle(segments)
    
            for seg_path in segments:
    
                # -----------------------------
                # 1. build segment
                # -----------------------------
                segment = Comma_Segment(seg_path, self.target_size)
    
                # -----------------------------
                # 2. load frames
                # -----------------------------
                frames = self.base._load_segment_frames(segment)
    
                # -----------------------------
                # 3. use YOUR existing trajectory class
                # -----------------------------
                traj = LeRobotTrajectory(
                    segment=segment,
                    frames=frames,
                    future_time=self.future_time
                )
    
                # -----------------------------
                # 4. build tensors using trajectory
                # -----------------------------
                images = []
                states = []
                actions = []
    
                for i in range(len(traj)):
                    obs, act = traj.get_step(i)
    
                    images.append(obs["image"])
                    states.append(obs["state"])
                    actions.append(act)
    
                # -----------------------------
                # 5. return LeRobot-style sample
                # -----------------------------
                yield {
                    "observations": {
                        "image": torch.stack(images),   # (T, 3, H, W)
                        "state": torch.stack(states)    # (T, 2)
                    },
                    "actions": torch.stack(actions),   # (T, 2)
                    "length": len(traj)
                }

if __name__ == "__main__":
    chunk = Path("comma2k19_data/extracted/Chunk_1")
    dataset = Comma_Instance(chunk, target_size=(256, 256), future_time=1.0)
    loader = DataLoader(dataset, batch_size=32, num_workers=0)

    for batch in loader:
        print(batch["x_frame"].shape)     # (32, 3, 256, 256)
        print(batch["y_speed"].shape)     # (32,)

        # testing frame
        frame = batch["x_frame"][0]
        frame = frame.permute(1, 2, 0).numpy()         
        frame = (frame * 255).astype(np.uint8)          

        cv2.imshow("test frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break