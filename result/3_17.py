import numpy as np
from scipy import signal
import os

# Load the original data
data = np.load('data/17_original.npy')

# Compute PSD using Welch's method
fs = 250  # Assuming sampling rate is 250Hz (common for EEG)
nperseg = 4 * fs  # 4-second window
noverlap = nperseg // 2  # 50% overlap
frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap)

# Reshape to (7, x) format
psd_reshaped = psd.reshape(7, -1)

# Create result directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Save the results
np.save('result/3_17.npy', psd_reshaped)