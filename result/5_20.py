import numpy as np
from antropy import sample_entropy, app_entropy, spectral_entropy
import os

data = np.load('data/20_original.npy')
n_channels, n_samples = data.shape
fs = 250
samples_per_10s = 10 * fs
n_segments = n_samples // samples_per_10s
results = np.zeros((7, 3, n_segments))

for seg in range(n_segments):
    start = seg * samples_per_10s
    end = start + samples_per_10s
    segment = data[:, start:end]
    
    for ch in range(7):
        signal = segment[ch]
        results[ch, 0, seg] = sample_entropy(signal)
        results[ch, 1, seg] = app_entropy(signal)
        results[ch, 2, seg] = spectral_entropy(signal, sf=fs, method='welch')

os.makedirs('result', exist_ok=True)
np.save('result/5_20.npy', results)