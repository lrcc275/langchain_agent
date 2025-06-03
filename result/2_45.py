
import numpy as np
from scipy import signal
import os

# Create result directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Load data
data = np.load('data/45_original.npy')
n_channels, n_samples = data.shape

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
window_size = 30 * fs  # 30 seconds
step_size = 10 * fs    # 10 seconds sliding window

# Frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Function to calculate band power
def bandpower(data, fs, band):
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024)
    idx = np.where((f >= band[0]) & (f <= band[1]))[0]
    return np.trapz(Pxx[idx], f[idx])

# Process data with sliding window
results = []
for start in range(0, n_samples - window_size + 1, step_size):
    segment = data[:, start:start + window_size]
    segment_result = []
    for channel in segment:
        channel_powers = []
        for band_name, band_range in bands.items():
            power = bandpower(channel, fs, band_range)
            channel_powers.append(power)
        segment_result.append(channel_powers)
    results.append(segment_result)

# Convert to numpy array and reshape
results_array = np.array(results)
n_windows, n_channels, n_bands = results_array.shape
final_shape = (n_windows, 7, 4)  # Assuming 7 channels, 4 bands
results_array = results_array.reshape(final_shape)

# Save results
np.save('result/2_45.npy', results_array)
