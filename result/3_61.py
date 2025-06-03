import numpy as np
from scipy import signal

# Load data
data = np.load('data/61_original.npy')

# Compute PSD for each channel using Welch method
fs = 250  # Assuming sampling rate is 250Hz (common for EEG)
nperseg = 4 * fs  # 4-second window
noverlap = nperseg // 2  # 50% overlap
frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap)

# Print results
print("PSD results:")
print(psd)

# Ensure shape is (7, x) and save
if psd.shape[0] != 7:
    psd = psd.T  # Transpose if needed
np.save('result/3_61.npy', psd)

print(f"Saved PSD data with shape {psd.shape} to result/3_61.npy")