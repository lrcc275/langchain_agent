import numpy as np
from scipy import signal

# Load data
data = np.load('data/5_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250 Hz (adjust if needed)
window_size = 4 * fs  # 4-second window
noverlap = window_size // 2  # 50% overlap

# Compute PSD for each channel
psd_results = []
for channel in data:
    f, psd = signal.welch(channel, fs=fs, nperseg=window_size, noverlap=noverlap)
    psd_results.append(psd)

# Convert to numpy array and reshape to (7, x)
psd_results = np.array(psd_results)
print(psd_results)

# Save to result/3_5.npy
np.save('result/3_5.npy', psd_results)