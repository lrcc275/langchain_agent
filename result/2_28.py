import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/28_original.npy')
fs = 250  # Assuming sampling rate is 250Hz

# Define frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Parameters
window_length = 30 * fs  # 30 seconds
slide_length = 10 * fs   # 10 seconds
n_channels = data.shape[0]  # Note: channels are first dimension in this data
total_samples = data.shape[1]

# Initialize results list
results = []

# Process with sliding window
for start in range(0, total_samples - window_length + 1, slide_length):
    segment = data[:, start:start + window_length]  # Note: channels first
    channel_powers = []
    
    for ch in range(n_channels):
        # Compute PSD using Welch's method
        f, psd = signal.welch(segment[ch], fs=fs, nperseg=1024)
        
        band_powers = []
        for band, (low, high) in bands.items():
            # Find indices of frequencies in band
            idx = np.logical_and(f >= low, f <= high)
            # Compute average power in band
            band_power = np.mean(psd[idx])
            band_powers.append(band_power)
        
        channel_powers.append(band_powers)
    
    results.append(channel_powers)

# Convert to numpy array (n_windows, n_channels, n_bands)
results_array = np.array(results)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/2_28.npy', results_array)
