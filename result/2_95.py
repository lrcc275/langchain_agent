import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/95_original.npy')
fs = 250  # Assuming sampling rate is 250Hz

# Define frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Parameters
window_length = 30 * fs  # 30s in samples
slide_length = 10 * fs   # 10s in samples
n_channels = data.shape[0]

# Prepare output array
n_windows = (data.shape[1] - window_length) // slide_length + 1
result = np.zeros((n_windows, n_channels, 4))  # (x, 7, 4) array

# Calculate band powers for each window
for i in range(n_windows):
    start = i * slide_length
    end = start + window_length
    window_data = data[:, start:end]
    
    for ch in range(n_channels):
        f, Pxx = signal.welch(window_data[ch], fs, nperseg=1024)
        for band_idx, (band, (low, high)) in enumerate(bands.items()):
            mask = (f >= low) & (f <= high)
            result[i, ch, band_idx] = np.sum(Pxx[mask])

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/2_95.npy', result)