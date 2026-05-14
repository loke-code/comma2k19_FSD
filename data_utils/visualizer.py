import cv2
import numpy as np
from pathlib import Path

def video_overlay(segment_path):
    segment = Path(segment_path)

    # frame_time -> absolute time anchor (exact time of incident light on camera lens 
    # (avoids camera CPU processing time))
    frame_times = np.load(segment / "global_pose" / "frame_times").flatten()
    
    # timestamps of speed and steer
    steer_t = np.load(segment / "processed_log" / "CAN" / "steering_angle" / "t").flatten()
    speed_t = np.load(segment / "processed_log" / "CAN" / "speed" / "t").flatten()
    # corresponding values
    steer_val = np.load(segment / "processed_log" / "CAN" / "steering_angle" / "value").flatten()
    speed_val = np.load(segment / "processed_log" / "CAN" / "speed" / "value").flatten()

    video_path = str(segment / "video.hevc")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # frame by frame overlay
    for i, f_time in enumerate(frame_times):
        ret, frame = cap.read()
        if not ret:
            break

        # interpolation, find closest speed and steer time, and interpolate the difference
        current_speed = np.interp(f_time, speed_t, speed_val)
        current_steer = np.interp(f_time, steer_t, steer_val)

        # drop width to get perfect square frame
        h, w, _ = frame.shape  # h=874, w=1164
        start_x = (w - h) // 2 # (1164 - 874) / 2 = 145
        frame = frame[:, start_x : start_x + h]
        #frame = cv2.resize(frame, (252, 252))

        # text overlay
        cv2.putText(frame, f"Speed: {current_speed:.1f} m/s", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Steer: {current_steer:.1f} deg", (50, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # visual steering stick
        cx, cy = int(874 / 2), 800  
        line_length = 200
        display_angle = np.radians(current_steer) 
        end_x = int(cx - line_length * np.sin(display_angle))
        end_y = int(cy - line_length * np.cos(display_angle))
        cv2.line(frame, (cx, cy), (end_x, end_y), (0, 0, 255), 8)
        

        cv2.imshow("Comma2k19 Viewer - Ground Truth", frame)

        # exit 'q'
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # run
    # update path as needed
    test_path = r"comma2k19_data\extracted\Chunk_1\b0c9d2329ad1606b_2018-07-27--06-03-57\7"
    video_overlay(test_path)