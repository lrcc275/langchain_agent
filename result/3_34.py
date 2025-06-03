import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/34_original.npy')

# Compute PSD for each channel
fs = 250  # Assuming sampling rate is 250Hz
nperseg = 4 * fs  # 4-second window
noverlap = nperseg // 2  # 50% overlap
frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/3_34.npy', psd)