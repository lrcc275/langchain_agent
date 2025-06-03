
import numpy as np
from scipy import signal
import os

# Load the original data
data = np.load('data/12_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
window_size = 30 * fs  # 30 seconds in samples
step_size = 10 * fs    # 10 seconds in samples
n_channels = data.shape[0]

# Band frequencies (Hz)
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Function to calculate band power
def bandpower(data, sf, band):
    band = np.asarray(band)
    freqs, psd = signal.welch(data, sf, nperseg=4*sf)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[idx_band]) * freq_res

# Process data with sliding window
results = []
for start in range(0, data.shape[1] - window_size + 1, step_size):
    window = data[:, start:start + window_size]
    window_results = []
    for ch in range(n_channels):
        channel_powers = []
        for band_name, band_range in bands.items():
            power = bandpower(window[ch], fs, band_range)
            channel_powers.append(power)
        window_results.append(channel_powers)
    results.append(window_results)

# Convert to numpy array (n_windows, n_channels, n_bands)
results_array = np.array(results)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/2_12.npy', results_array)
