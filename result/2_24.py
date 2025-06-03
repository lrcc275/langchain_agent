
import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/24_original.npy')
fs = 250  # Assuming sampling rate is 250Hz

# Define frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Parameters
window_size = 30 * fs  # 30 seconds
step_size = 10 * fs    # 10 seconds sliding window
n_channels = data.shape[0]
n_bands = len(bands)

# Initialize result array
results = []

# Process each window
for start in range(0, data.shape[1] - window_size + 1, step_size):
    window = data[:, start:start + window_size]
    band_powers = np.zeros((n_channels, n_bands))
    
    for ch in range(n_channels):
        # Compute PSD using Welch's method
        f, psd = signal.welch(window[ch], fs, nperseg=1024)
        
        # Calculate band powers
        for i, (band, (low, high)) in enumerate(bands.items()):
            mask = (f >= low) & (f <= high)
            band_powers[ch, i] = np.sum(psd[mask])
    
    results.append(band_powers)

# Convert to numpy array
results_array = np.array(results)

# Print results
print("Band powers for each window:")
print(results_array)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/2_24.npy', results_array)
