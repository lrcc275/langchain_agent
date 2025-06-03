import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/57_original.npy')
fs = 250  # Assuming sampling rate is 250Hz

# Define parameters
window_length = 30 * fs  # 30 seconds
step_size = 10 * fs      # 10 seconds overlap
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
    idx = np.where((f >= band[0]) & (f <= band[1]))[0]
    return np.mean(Pxx[idx])

# Process data with sliding window
results = []
for start in range(0, data.shape[1] - window_length + 1, step_size):
    window = data[:, start:start+window_length]
    window_result = []
    for ch in range(n_channels):
        channel_powers = []
        for band_name, band_range in bands.items():
            power = bandpower(window[ch], fs, band_range)
            channel_powers.append(power)
        window_result.append(channel_powers)
    results.append(window_result)

# Convert to numpy array (n_windows, n_channels, n_bands)
results_array = np.array(results)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/2_57.npy', results_array)
