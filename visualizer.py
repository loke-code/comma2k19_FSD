import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def video_overlay(segment_path):
    segment = Path(segment_path)

    # timestamps
    frame_times = np.load(segment / "global_pose" / "frame_times").flatten()

    steer_t = np.load(segment / "processed_log" / "CAN" / "steering_angle" / "t").flatten()
    speed_t = np.load(segment / "processed_log" / "CAN" / "speed" / "t").flatten()

    steer_val = np.load(segment / "processed_log" / "CAN" / "steering_angle" / "value").flatten()
    speed_val = np.load(segment / "processed_log" / "CAN" / "speed" / "value").flatten()

    video_path = str(segment / "video.hevc")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    for i, f_time in enumerate(frame_times):

        ret, frame = cap.read()
        if not ret:
            break

        # interpolate signals
        current_speed = np.interp(f_time, speed_t, speed_val)
        current_steer = np.interp(f_time, steer_t, steer_val)

        # text overlay
        cv2.putText(
            frame,
            f"Speed: {current_speed:.1f} km/hr",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Steer: {current_steer:.1f} deg",
            (50, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        # steering visualization
        h, w = frame.shape[:2]
        cx, cy = w // 2, h - 250

        line_length = 200
        angle = np.radians(current_steer)

        end_x = int(cx - line_length * np.sin(angle))
        end_y = int(cy - line_length * np.cos(angle))

        cv2.line(frame, (cx, cy), (end_x, end_y), (0, 0, 255), 8)

        # convert for matplotlib (BGR -> RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # display frame
        plt.figure(figsize=(10, 6))
        plt.imshow(frame_rgb)
        plt.axis("off")

        display(plt.gcf())
        clear_output(wait=True)

        plt.pause(0.03)

    cap.release()