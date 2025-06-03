import numpy as np
from scipy import signal
import os

# Create result directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Load the EEG data
data = np.load('data/88_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
window_size = 30 * fs  # 30 seconds in samples
step_size = 10 * fs    # 10 seconds in samples
n_channels = data.shape[0]

# Frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Function to calculate band energy
def calculate_band_energy(signal_data, fs, band):
    nyq = 0.5 * fs
    low, high = band
    b, a = signal.butter(4, [low/nyq, high/nyq], btype='bandpass')
    filtered = signal.filtfilt(b, a, signal_data)
    return np.sum(filtered**2)

# Process data with sliding windows
results = []
num_windows = (data.shape[1] - window_size) // step_size + 1

for i in range(num_windows):
    start = i * step_size
    end = start + window_size
    window_data = data[:, start:end]
    
    window_result = []
    for ch in range(n_channels):
        channel_result = []
        for band_name, band_range in bands.items():
            energy = calculate_band_energy(window_data[ch], fs, band_range)
            channel_result.append(energy)
        window_result.append(channel_result)
    results.append(window_result)

# Convert to numpy array and save
results_array = np.array(results)
np.save('result/2_88.npy', results_array)
