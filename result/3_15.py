import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/15_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz (common for EEG)
nperseg = 4 * fs  # 4 second window
noverlap = nperseg // 2  # 50% overlap

# Compute PSD for each channel
psd_results = []
for channel in data:
    f, Pxx = signal.welch(channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
    psd_results.append(Pxx)

# Convert to numpy array and save
psd_results = np.array(psd_results)
os.makedirs('result', exist_ok=True)
np.save('result/3_15.npy', psd_results)