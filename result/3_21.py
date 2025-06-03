
import numpy as np
from scipy import signal

# Load the data
data = np.load('data/21_original.npy')

# Parameters
fs = 250  # Assuming a sampling rate of 250Hz (common for EEG)
nperseg = 4 * fs  # 4-second window
noverlap = nperseg // 2  # 50% overlap

# Compute PSD for each channel
psd_results = []
for channel in data:
    f, Pxx = signal.welch(channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
    psd_results.append(Pxx)

# Convert to numpy array and reshape to (7, x)
psd_array = np.array(psd_results)
if len(psd_array.shape) == 1:
    psd_array = psd_array.reshape(7, -1)

# Save the results
np.save('result/3_21.npy', psd_array)
