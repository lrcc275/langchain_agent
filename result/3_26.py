import numpy as np
from scipy import signal

# Load data
data = np.load("data/26_original.npy")

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
nperseg = 4 * fs  # 4-second window
noverlap = nperseg // 2  # 50% overlap

# Compute PSD for each channel
psd_results = []
for channel in data:
    f, Pxx = signal.welch(channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
    psd_results.append(Pxx)

# Convert to numpy array
psd_array = np.array(psd_results)
print(psd_array)

# Save results
np.save("result/3_26.npy", psd_array)
