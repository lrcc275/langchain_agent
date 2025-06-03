import numpy as np
from scipy import signal
import os

# Create result directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Load data
data = np.load('data/61_original.npy')
fs = 250  # Assuming sampling rate is 250Hz

# Define parameters
window_length = 30 * fs  # 30 seconds
slide_length = 10 * fs   # 10 seconds
n_channels = data.shape[0]

# Define frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Function to calculate band power
def bandpower(data, fs, band):
    f, Pxx = signal.welch(data, fs=fs, nperseg=fs*2)
    band_mask = (f >= band[0]) & (f < band[1])
    return np.mean(Pxx[band_mask])

# Process data with sliding windows
results = []
for start in range(0, data.shape[1] - window_length + 1, slide_length):
    window = data[:, start:start + window_length]
    window_result = []
    for channel in range(n_channels):
        channel_powers = []
        for band_name, band_range in bands.items():
            power = bandpower(window[channel], fs, band_range)
            channel_powers.append(power)
        window_result.append(channel_powers)
    results.append(window_result)

# Convert to numpy array (n_windows, n_channels, n_bands)
results_array = np.array(results)

# Save results
np.save('result/2_61.npy', results_array)