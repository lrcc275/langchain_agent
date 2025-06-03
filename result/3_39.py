import numpy as np
from scipy import signal

data = np.load('data/39_original.npy')
fs = 250
window_size = 4 * fs
noverlap = window_size // 2

psd_results = []
for channel in data:
    f, Pxx = signal.welch(channel, fs, nperseg=window_size, noverlap=noverlap)
    psd_results.append(Pxx)

np.save('result/3_39.npy', np.array(psd_results))