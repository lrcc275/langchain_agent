import numpy as np
from scipy import signal
import os

# Load the original data
data = np.load('data/52_original.npy')

# Calculate PSD using Welch's method
fs = 250  # Assuming sampling rate is 250Hz (common for EEG)
nperseg = 4 * fs  # 4-second window
noverlap = nperseg // 2  # 50% overlap
frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap)

# Print the results
print("PSD results:")
print(psd)

# Ensure result directory exists
os.makedirs('result', exist_ok=True)

# Save the PSD results (7 channels x frequency bins)
np.save('result/3_52.npy', psd)