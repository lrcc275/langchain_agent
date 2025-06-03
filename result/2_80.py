
import numpy as np
from scipy.signal import welch
import os

# Load the original data
data = np.load('data/80_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
window_size = 30 * fs  # 30 seconds
step_size = 10 * fs    # 10 seconds
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Create result directory if not exists
os.makedirs('result', exist_ok=True)

# Process data
results = []
for start in range(0, data.shape[1] - window_size + 1, step_size):
    segment = data[:, start:start + window_size]
    band_powers = []
    
    for channel in segment:
        freqs, psd = welch(channel, fs=fs, nperseg=min(1024, len(channel)))
        channel_bands = []
        for band, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            channel_bands.append(np.sum(psd[band_mask]))
        band_powers.append(channel_bands)
    
    results.append(band_powers)

# Convert to numpy array (n_windows, n_channels, 4_bands)
results_array = np.array(results)

# Save results
np.save('result/2_80.npy', results_array)
