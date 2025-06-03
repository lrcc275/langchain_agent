import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/72_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz (common for EEG)
window_size = 4 * fs  # 4 seconds in samples
noverlap = window_size // 2  # 50% overlap

# Compute PSD for each channel
psd_results = []
for channel in data:
    f, Pxx = signal.welch(channel, fs=fs, nperseg=window_size, noverlap=noverlap)
    psd_results.append(Pxx)

# Convert to numpy array and reshape to (7, x)
psd_array = np.array(psd_results)
if psd_array.ndim == 1:
    psd_array = psd_array.reshape(7, -1)

# Print results
print("PSD results:")
print(psd_array)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/3_72.npy', psd_array)