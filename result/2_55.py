import numpy as np
from scipy.signal import welch
import os

# Load data
data = np.load('data/55_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
window_size = 30 * fs  # 30 seconds
slide_size = 10 * fs  # 10 seconds
n_channels = data.shape[0]

# Band definitions
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Initialize result list
results = []

# Process with sliding windows
for start in range(0, data.shape[1] - window_size + 1, slide_size):
    segment = data[:, start:start + window_size]
    segment_powers = np.zeros((n_channels, len(bands)))
    
    for ch in range(n_channels):
        f, Pxx = welch(segment[ch], fs=fs, nperseg=min(1024, window_size))
        
        for i, (band, (low, high)) in enumerate(bands.items()):
            mask = (f >= low) & (f <= high)
            segment_powers[ch, i] = np.sum(Pxx[mask])
    
    results.append(segment_powers)

# Convert to numpy array (n_windows, n_channels, n_bands)
result_array = np.array(results)

# Save results
np.save('result/2_55.npy', result_array)