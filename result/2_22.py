import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/22_original.npy')  # Shape is (7, 137445) - channels x time

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
window_size = 30 * fs  # 30 seconds in samples
step_size = 10 * fs  # 10 seconds sliding window in samples
n_channels = data.shape[0]
total_samples = data.shape[1]

# Frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Function to calculate band power
def bandpower(data, fs, band):
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024)
    idx = np.where((f >= band[0]) & (f <= band[1]))[0]
    return np.trapz(Pxx[idx], f[idx])

# Process data with sliding window
results = []
for start in range(0, total_samples - window_size + 1, step_size):
    window = data[:, start:start + window_size]  # Take window across all channels
    channel_powers = []
    for ch in range(n_channels):
        band_powers = []
        for band_name, band_range in bands.items():
            power = bandpower(window[ch], fs, band_range)
            band_powers.append(power)
        channel_powers.append(band_powers)
    results.append(channel_powers)

# Convert to numpy array and save
results_array = np.array(results)
print("Results shape:", results_array.shape)
print("First window results:", results_array[0])

# Save the results
os.makedirs('result', exist_ok=True)
np.save('result/2_22.npy', results_array)