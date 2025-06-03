import numpy as np
from scipy import signal
import os

data = np.load('data/72_original.npy')
fs = 250

window_length = 30 * fs
slide_length = 10 * fs
num_channels = data.shape[0]

bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

def bandpower(data, fs, band):
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024)
    idx_band = np.logical_and(f >= band[0], f < band[1])
    return np.trapz(Pxx[idx_band], f[idx_band])

results = []
num_windows = (data.shape[1] - window_length) // slide_length + 1

for i in range(num_windows):
    start = i * slide_length
    end = start + window_length
    window = data[:, start:end]
    
    window_result = []
    for ch in range(num_channels):
        channel_powers = []
        for band_name, band_range in bands.items():
            power = bandpower(window[ch], fs, band_range)
            channel_powers.append(power)
        window_result.append(channel_powers)
    
    results.append(window_result)

results_array = np.array(results)
np.save('result/2_72.npy', results_array)