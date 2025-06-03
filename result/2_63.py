import numpy as np
from scipy import signal
import os

# Create result directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Load data
data = np.load('data/63_original.npy')
n_channels = data.shape[0]
fs = 250  # Assuming sampling rate of 250Hz

# Define frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Parameters
window_size = 30 * fs  # 30 seconds in samples
step_size = 10 * fs    # 10 seconds in samples
n_windows = (data.shape[1] - window_size) // step_size + 1

# Initialize result array (windows x channels x bands)
result = np.zeros((n_windows, n_channels, 4))

# Process each window
for win_idx in range(n_windows):
    start = win_idx * step_size
    end = start + window_size
    window_data = data[:, start:end]
    
    # Calculate PSD using Welch's method
    for ch_idx in range(n_channels):
        f, psd = signal.welch(window_data[ch_idx], fs=fs, nperseg=1024)
        
        # Calculate band powers
        for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
            mask = (f >= low) & (f < high)
            result[win_idx, ch_idx, band_idx] = np.sum(psd[mask])

# Save results
np.save('result/2_63.npy', result)
