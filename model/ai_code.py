import numpy as np
import time
import warnings
import threading
import pandas as pd
from scipy.signal import butter, filtfilt
from brainaccess import core
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.core.gain_mode import GainMode
import queue
from tensorflow.keras.models import load_model
from scipy import stats
from collections import deque
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Manager

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def _acq_closure(ch_number: int = 1, data_queue=None):
    mutex = threading.Lock()

    def _acq_callback(chunk, chunk_size):
        nonlocal mutex
        with mutex:
            if data_queue is not None:
                data_queue.put((chunk.copy(), chunk_size))

    def get_data():
        raise NotImplementedError("get_data is not used when using data_queue.")

    return _acq_callback, get_data

if __name__ == "__main__":
    device_name = "BA MINI 012"  # Change to your device name
    core.init()

    # Load your pre-trained model
    model_path = 'eeg_classification_model.h5'  # Adjust the path if necessary
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")

    # Load label encoder used during training
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['baseline', 'ciekawe', 'nuda'])  # Adjust according to your labels

    # Scan for devices and find the specified one
    core.scan(0)  # Adapter number (on Windows always 0)
    count = core.get_device_count()
    port = 0

    print("Found devices:", count)
    for i in range(count):
        name = core.get_device_name(i)
        if device_name in name:
            port = i
            
         # Initialize Manager for sharing data
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['predicted_label'] = None       
            

    # Connect to the device
    with EEGManager() as mgr:
        print("Connecting to device:", core.get_device_name(port))
        _status = mgr.connect(port)
        if _status == 1:
            raise Exception("Connection failed")
        elif _status == 2:
            warnings.warn("Stream is incompatible. Update the firmware.")

        # Battery info
        print("Battery level:", mgr.get_battery_info().level)

        # Set the channels
        print("Setting the channels")
        eeg_channels_number = 8  # Adjust based on your device
        ch_nr = eeg_channels_number
        for i in range(eeg_channels_number):
            mgr.set_channel_enabled(eeg_channel.ELECTRODE_MEASUREMENT + i, True)
            mgr.set_channel_gain(eeg_channel.ELECTRODE_MEASUREMENT + i, GainMode.X8)
            mgr.set_channel_bias(eeg_channel.ELECTRODE_MEASUREMENT + i, True)

        # Check if the device has an accelerometer
        has_accel = mgr.get_device_features().has_accel()
        if has_accel:
            print("Setting the accelerometer")
            mgr.set_channel_enabled(eeg_channel.ACCELEROMETER, True)
            mgr.set_channel_enabled(eeg_channel.ACCELEROMETER + 1, True)
            mgr.set_channel_enabled(eeg_channel.ACCELEROMETER + 2, True)
            ch_nr += 3

        mgr.set_channel_enabled(eeg_channel.SAMPLE_NUMBER, True)
        ch_nr += 1

        mgr.set_channel_enabled(eeg_channel.STREAMING, True)
        ch_nr += 1

        # Set the sample rate
        sr = mgr.get_sample_frequency()
        print("Sample rate:", sr)

        # Define window size and processing interval
        window_size = 7  # in seconds, matches your training window size
        processing_interval = 5  # in seconds

        window_size_samples = int(window_size * sr)
        data_buffer = deque(maxlen=window_size_samples)

        # Initialize the queue for thread-safe data transfer
        data_queue = queue.Queue()

        # Define the callback for the acquisition
        _acq_callback, _ = _acq_closure(
            ch_number=ch_nr, data_queue=data_queue
        )
        mgr.set_callback_chunk(_acq_callback)

        # Load defined configuration
        mgr.load_config()

        # Start the stream
        mgr.start_stream()
        print("Stream started")

        # Prepare for real-time processing
        ch = []
        ch.append("sample")
        ch.extend([f"ch_{i+1}" for i in range(eeg_channels_number)])
        if has_accel:
            ch.extend(["accel_x", "accel_y", "accel_z"])
        ch.extend(["streaming"])

        # Initialize the CSV file with headers
        csv_file = 'amelka_uwu_dobra.csv'
        with open(csv_file, 'w') as f:
            f.write(','.join(ch) + '\n')

        try:
            print("\nStarting real-time analysis. Press Ctrl+C to stop.")
            while True:
                time.sleep(processing_interval)
                chunks = []
                while not data_queue.empty():
                    chunk, chunk_size = data_queue.get()
                    chunks.append(chunk)
                    data_queue.task_done()
                if chunks:
                    # Concatenate chunks
                    dat = np.concatenate(chunks, axis=1)
                    # Create DataFrame
                    df = pd.DataFrame(dat.T, columns=ch)

                    # Apply bandpass filter (1-50 Hz)
                    df.iloc[:, 1:eeg_channels_number+1] = df.iloc[:, 1:eeg_channels_number+1].apply(
                        lambda x: bandpass_filter(x, 1, 50, sr), axis=0
                    )

                    # Save the DataFrame to CSV in append mode
                    df.to_csv(csv_file, mode='a', header=False, index=False)

                    # Append new data to buffer
                    eeg_data = df.iloc[:, 1:eeg_channels_number+1].values  # shape: (n_samples_chunk, n_channels)
                    for row in eeg_data:
                        data_buffer.append(row)

                    # If buffer has enough data
                    if len(data_buffer) == window_size_samples:
                        # Prepare data for the model
                        buffer_array = np.array(data_buffer)  # shape: (window_size_samples, n_channels)

                        # Remove outliers using Z-score (per sample)
                        z_scores = np.abs(stats.zscore(buffer_array, axis=0))
                        buffer_array[z_scores > 3] = np.nan

                        # Handle missing values by interpolation
                        buffer_df = pd.DataFrame(buffer_array)
                        buffer_df = buffer_df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                        buffer_array = buffer_df.values

                        # Transpose to (n_channels, window_size_samples)
                        buffer_array = buffer_array.T  # shape: (n_channels, window_size_samples)

                        # Filter signals for each channel
                        filtered_alpha = []
                        filtered_beta = []

                        for channel_data in buffer_array:
                            # Filter the signals
                            alpha = bandpass_filter(channel_data, 8.0, 12.0, sr)
                            beta = bandpass_filter(channel_data, 13.0, 30.0, sr)
                            filtered_alpha.append(alpha)
                            filtered_beta.append(beta)

                        # Convert to numpy arrays
                        filtered_alpha = np.array(filtered_alpha)  # shape: (n_channels, window_size_samples)
                        filtered_beta = np.array(filtered_beta)    # shape: (n_channels, window_size_samples)

                        # Stack alpha and beta bands along the last dimension
                        segment = np.stack([filtered_alpha, filtered_beta], axis=-1)  # shape: (n_channels, window_size_samples, 2)

                        # Reshape data to match model's expected input
                        # Transpose to (window_size_samples, n_channels, 2)
                        segment_reshaped = segment.transpose(1, 0, 2)  # shape: (window_size_samples, n_channels, 2)
                        # Reshape to (1, window_size_samples, n_channels * 2)
                        segment_reshaped = segment_reshaped.reshape(1, segment_reshaped.shape[0], -1)

                        # Debugging: Print shapes
                        print(f"Segment reshaped shape: {segment_reshaped.shape}")

                        # Make prediction
                        prediction = model.predict(segment_reshaped)
                        predicted_label_index = np.argmax(prediction, axis=-1)[0]
                        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

                        # Print the predicted label
                        
                        shared_dict['predicted_label'] = predicted_label
                        print(f"Predicted label: {predicted_label}")

        except KeyboardInterrupt:
            # Stop the stream gracefully
            mgr.stop_stream()
            print("\nStream stopped by user.")

    # Close the core
    core.close()