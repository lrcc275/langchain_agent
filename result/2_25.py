
import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/25_original.npy')
fs = 250  # Assuming sampling rate is 250Hz

# Define frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Parameters
window_length = 30 * fs  # 30 seconds in samples
slide_length = 10 * fs    # 10 seconds in samples

# Initialize result array
num_windows = (data.shape[1] - window_length) // slide_length + 1
result = np.zeros((num_windows, data.shape[0], len(bands)))

# Calculate band powers for each window
for i in range(num_windows):
    start = i * slide_length
    end = start + window_length
    window_data = data[:, start:end]
    
    for ch in range(data.shape[0]):
        f, Pxx = signal.welch(window_data[ch], fs=fs, nperseg=1024)
        
        for j, (band, (low, high)) in enumerate(bands.items()):
            mask = (f >= low) & (f <= high)
            result[i, ch, j] = np.sum(Pxx[mask])

# Save results
np.save('result/2_25.npy', result)
