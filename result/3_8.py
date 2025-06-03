
import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/8_original.npy')

# Parameters
fs = 250  # Assuming a sampling rate of 250 Hz (common for EEG)
window_size = 4 * fs  # 4 seconds window
noverlap = window_size // 2  # 50% overlap

# Compute PSD for each channel
psd_results = []
for channel in data:
    f, Pxx = signal.welch(channel, fs, nperseg=window_size, noverlap=noverlap)
    psd_results.append(Pxx)

# Convert to numpy array and reshape to (7, x)
psd_results = np.array(psd_results)
if psd_results.shape[0] != 7:
    psd_results = psd_results.reshape(7, -1)

# Save the results
os.makedirs('result', exist_ok=True)
np.save('result/3_8.npy', psd_results)

# Print the results
print(psd_results)
