import numpy as np
from scipy import signal
import os

# Load the original data
data = np.load('data/66_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
window_size = 30 * fs  # 30 seconds in samples
slide_size = 10 * fs    # 10 seconds in samples
num_channels = data.shape[0]

# Band definitions
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Function to calculate band power
def bandpower(data, fs, band):
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024)
    idx_band = np.logical_and(f >= band[0], f < band[1])
    return np.trapz(Pxx[idx_band], f[idx_band])

# Process data with sliding window
results = []
for start in range(0, data.shape[1] - window_size + 1, slide_size):
    segment = data[:, start:start + window_size]
    segment_powers = []
    for ch in range(num_channels):
        channel_powers = []
        for band_name, band_range in bands.items():
            power = bandpower(segment[ch], fs, band_range)
            channel_powers.append(power)
        segment_powers.append(channel_powers)
    results.append(segment_powers)

# Convert to numpy array (x, 7, 4)
results_array = np.array(results)

# Create result directory if not exists
os.makedirs('result', exist_ok=True)

# Save results
np.save('result/2_66.npy', results_array)
