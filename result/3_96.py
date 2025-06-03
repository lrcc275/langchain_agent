import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/96_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz (common for EEG)
window_size = 4 * fs  # 4 seconds window
noverlap = window_size // 2  # 50% overlap

# Calculate PSD for each channel
psd_results = []
for channel in data:
    f, Pxx = signal.welch(channel, fs=fs, nperseg=window_size, noverlap=noverlap)
    psd_results.append(Pxx)

# Convert to numpy array and reshape to (7, x)
psd_array = np.array(psd_results)
print(psd_array)

# Save results
np.save('result/3_96.npy', psd_array)