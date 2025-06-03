import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/20_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
nperseg = 4 * fs  # 4 second window
noverlap = nperseg // 2  # 50% overlap

# Calculate PSD for each channel
psd_results = []
for channel in data:
    f, Pxx = signal.welch(channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
    psd_results.append(Pxx)

psd_array = np.array(psd_results)

# Print results
print("PSD results for each channel:")
print(psd_array)

# Ensure result directory exists
os.makedirs('result', exist_ok=True)

# Save results
np.save('result/3_20.npy', psd_array)