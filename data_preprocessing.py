import numpy as np  
from scipy.interpolate import interp1d  
import sys  
import os  
  
# Add parent directory to path for openpilot tools  
sys.path.append('..')  
from tools.lib.framereader import FrameReader  
  
def sample_at_lowest_frequency(example_segment):  
    """  
    Sample velocity, steering angle, and accelerometer at the lowest frequency,  
    then sample video frames at that synchronized rate.  
      
    Args:  
        example_segment: Path to the segment directory containing the data  
          
    Returns:  
        Dictionary containing synchronized data at the lowest frequency  
    """  
    # Load video frame timestamps  
    frame_times = np.load(example_segment + 'global_pose/frame_times')  
      
    # Load velocity data (CAN speed)  
    vel_t = np.load(example_segment + 'processed_log/CAN/speed/t')  
    vel_val = np.load(example_segment + 'processed_log/CAN/speed/value')  
      
    # Load steering angle data (CAN)  
    steer_t = np.load(example_segment + 'processed_log/CAN/steering_angle/t')  
    steer_val = np.load(example_segment + 'processed_log/CAN/steering_angle/value')  
      
    # Load accelerometer data (IMU)  
    accel_t = np.load(example_segment + 'processed_log/IMU/acceleration/t')  
    accel_val = np.load(example_segment + 'processed_log/IMU/acceleration/value')  
      
    # Calculate sampling rates  
    vel_rate = len(vel_t) / (vel_t[-1] - vel_t[0])  
    steer_rate = len(steer_t) / (steer_t[-1] - steer_t[0])  
    accel_rate = len(accel_t) / (accel_t[-1] - accel_t[0])  
      
    print(f"Velocity rate: {vel_rate:.2f} Hz")  
    print(f"Steering angle rate: {steer_rate:.2f} Hz")  
    print(f"Accelerometer rate: {accel_rate:.2f} Hz")  
      
    # Determine lowest frequency (likely CAN data)  
    rates = {'velocity': vel_rate, 'steering': steer_rate, 'accelerometer': accel_rate}  
    lowest_sensor = min(rates, key=rates.get)  
    lowest_rate = rates[lowest_sensor]  
      
    print(f"Lowest frequency: {lowest_rate:.2f} Hz ({lowest_sensor})")  
      
    # Use the timestamp array of the lowest frequency sensor as the reference  
    if lowest_sensor == 'velocity':  
        ref_t = vel_t  
    elif lowest_sensor == 'steering':  
        ref_t = steer_t  
    else:  
        ref_t = accel_t  
      
    # Interpolate all sensor data to the reference timestamps  
    vel_at_ref = interp1d(vel_t, vel_val, bounds_error=False, fill_value="extrapolate")(ref_t)  
    steer_at_ref = interp1d(steer_t, steer_val, bounds_error=False, fill_value="extrapolate")(ref_t)  
    accel_at_ref = interp1d(accel_t, accel_val, bounds_error=False, fill_value="extrapolate")(ref_t)  
      
    # Create mapping from boot time to frame index  
    frame_index_from_time = interp1d(  
        frame_times,   
        np.arange(len(frame_times)),  
        bounds_error=False,   
        fill_value="extrapolate"  
    )  
      
    # Get frame indices for reference timestamps  
    frame_indices = frame_index_from_time(ref_t).astype(int)  
      
    # Initialize FrameReader  
    fr = FrameReader(example_segment + 'video.hevc')  
      
    # Extract frames at reference timestamps  
    frames = []  
    for idx in frame_indices:  
        if 0 <= idx < len(frame_times):  # Ensure valid index  
            frame = fr.get(idx, pix_fmt='rgb24')[0]  
            frames.append(frame)  
      
    return {  
        'ref_t': ref_t,  
        'ref_rate': lowest_rate,  
        'ref_sensor': lowest_sensor,  
        'vel_at_ref': vel_at_ref,  
        'steer_at_ref': steer_at_ref,  
        'accel_at_ref': accel_at_ref,  
        'frame_indices': frame_indices,  
        'frames': frames  
    }  
  
  
if __name__ == "__main__":  
    # Example usage  
    example_segment = '../Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/'  
      
    print("Sampling all sensors at lowest frequency and extracting video...")  
    synced_data = sample_at_lowest_frequency(example_segment)  
    print(f"Synchronized to {synced_data['ref_rate']:.2f} Hz ({synced_data['ref_sensor']})")  
    print(f"Extracted {len(synced_data['frames'])} video frames")  
    print(f"Velocity samples: {len(synced_data['vel_at_ref'])}")  
    print(f"Steering angle samples: {len(synced_data['steer_at_ref'])}")  
    print(f"Accelerometer samples: {len(synced_data['accel_at_ref'])}")