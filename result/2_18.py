import numpy as np
from scipy import signal
import os

# Create result directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Load the data
data = np.load('data/18_original.npy')

# Assuming sampling rate is 250Hz (common for EEG)
fs = 250
window_size = 30 * fs  # 30 seconds
step_size = 10 * fs    # 10 seconds sliding window

# Define frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Initialize results list
results = []

# Process each window
for start in range(0, data.shape[1] - window_size + 1, step_size):
    window = data[:, start:start + window_size]
    band_powers = []
    
    # Calculate band powers for each channel
    for channel in window:
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(channel, fs=fs, nperseg=1024)
        
        # Calculate power in each band
        channel_bands = []
        for band, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs < high)
            channel_bands.append(np.sum(psd[band_mask]))
        
        band_powers.append(channel_bands)
    
    results.append(band_powers)

# Convert to numpy array (n_windows, n_channels, 4_bands)
results_array = np.array(results)

# Print results
print("Band powers for each window and channel:")
print(results_array)

# Save results
np.save('result/2_18.npy', results_array)