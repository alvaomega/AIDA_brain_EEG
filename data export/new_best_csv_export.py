import numpy as np
import time
import warnings
import threading
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from brainaccess import core
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.core.gain_mode import GainMode
import queue  # Import the queue module

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype="bandpass", output="sos")
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
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

def analyze_band_powers(eeg_data, sr):
    """
    Analyze EEG data to compute relative power in different frequency bands.
    """
    # Define EEG bands
    bands = {'delta': (1, 4),
             'theta': (4, 7),
             'alpha': (8, 13),
             'beta': (13, 30),
             'gamma': (30, 50)}

    psd_list = []
    for i in range(eeg_data.shape[1]):
        eeg_signal = eeg_data[:, i]
        freqs, psd = welch(eeg_signal, fs=sr, nperseg=sr*2)
        psd_list.append(psd)
    psd_array = np.array(psd_list)
    total_power = psd_array.sum(axis=1)

    relative_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        band_idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
        band_power = psd_array[:, band_idx].mean(axis=1)
        relative_power = band_power / total_power
        relative_powers[band_name] = relative_power

    return freqs, psd_array, relative_powers

if __name__ == "__main__":
    device_name = "BA MINI 012"  # change to your device name
    core.init()

    # Scan for devices and find the specified one
    core.scan(0)  # adapter number (on Windows always 0)
    count = core.get_device_count()
    port = 0

    print("Found devices:", count)
    for i in range(count):
        name = core.get_device_name(i)
        if device_name in name:
            port = i

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
        ch_nr = 0
        eeg_channels_number = 8  # Adjust based on your device
        for i in range(eeg_channels_number):
            mgr.set_channel_enabled(eeg_channel.ELECTRODE_MEASUREMENT + i, True)
            mgr.set_channel_gain(eeg_channel.ELECTRODE_MEASUREMENT + i, GainMode.X8)
            mgr.set_channel_bias(eeg_channel.ELECTRODE_MEASUREMENT + i, True)
            ch_nr += 1

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
        window_size = 10  # in seconds
        processing_interval = 1  # in seconds

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
            moving_average_band_powers = {band: [] for band in ['theta', 'alpha', 'beta', 'gamma']}
            max_moving_average_length = 10  # number of values in the moving average

            # Boredom detection thresholds for each band (these values should be calibrated)
            boredom_thresholds = {
                'theta': 0.2,
                'alpha': 0.5,
                'beta': 0.1,
                'gamma': 0.05
            }

            print("\nStarting real-time analysis with multi-band powers. Press Ctrl+C to stop.")
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
                    df.iloc[:, 1:eeg_channels_number+1] = butter_bandpass_filter(
                        df.iloc[:, 1:eeg_channels_number+1].T, 1, 50, sr
                    ).T

                    # Save the DataFrame to CSV in append mode
                    df.to_csv(csv_file, mode='a', header=False, index=False)

                    # Analyze band powers
                    eeg_data = df.iloc[:, 1:eeg_channels_number+1].values
                    freqs, psd_array, relative_powers = analyze_band_powers(eeg_data, sr)

                    # Compute average relative power across channels for each band
                    avg_relative_powers = {band: np.mean(power) for band, power in relative_powers.items()}

                    # Update moving averages
                    for band in moving_average_band_powers.keys():
                        moving_average_band_powers[band].append(avg_relative_powers[band])
                        if len(moving_average_band_powers[band]) > max_moving_average_length:
                            moving_average_band_powers[band].pop(0)

                    # Compute moving averages
                    moving_avgs = {band: np.mean(moving_average_band_powers[band]) for band in moving_average_band_powers.keys()}

                    # Print current moving averages
                    print("Current moving averages of relative band powers:")
                    for band in moving_avgs.keys():
                        print(f"  {band.capitalize()}: {moving_avgs[band]:.4f}")

                    # Boredom detection logic (example using alpha and theta bands)
                    if moving_avgs['alpha'] > boredom_thresholds['alpha'] and moving_avgs['theta'] > boredom_thresholds['theta']:
                        print("Alert: The person might be bored!")

        except KeyboardInterrupt:
            # Stop the stream gracefully
            mgr.stop_stream()
            print("\nStream stopped by user.")

    # Close the core
    core.close()