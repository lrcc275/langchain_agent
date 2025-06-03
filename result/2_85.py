
import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/85_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
window_size = 30 * fs  # 30 seconds
slide_size = 10 * fs  # 10 seconds
n_channels = data.shape[0]

# Band definitions (Hz)
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Function to calculate band power
def bandpower(data, fs, band):
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024)
    band_idx = np.where((f >= band[0]) & (f <= band[1]))[0]
    return np.mean(Pxx[band_idx])

# Process data
results = []
for start in range(0, data.shape[1] - window_size + 1, slide_size):
    segment = data[:, start:start+window_size]
    segment_powers = []
    for ch in range(n_channels):
        channel_powers = []
        for band_name, band_range in bands.items():
            power = bandpower(segment[ch], fs, band_range)
            channel_powers.append(power)
        segment_powers.append(channel_powers)
    results.append(segment_powers)

# Convert to numpy array (n_segments, n_channels, n_bands)
results_array = np.array(results)

# Print results
print("Band powers for each segment and channel:")
print(results_array)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/2_85.npy', results_array)
