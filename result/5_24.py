
import numpy as np
from antropy import sample_entropy, spectral_entropy, app_entropy
import os

data = np.load('data/24_original.npy')

fs = 250
segment_length = 10 * fs
n_segments = data.shape[1] // segment_length
n_channels = data.shape[0]

results = np.zeros((7, 3, n_segments))

for seg in range(n_segments):
    start = seg * segment_length
    end = start + segment_length
    for ch in range(7):
        segment = data[ch, start:end]
        sampen = sample_entropy(segment, order=2)
        apen = app_entropy(segment, order=2)
        spen = spectral_entropy(segment, sf=fs, method='fft')
        results[ch, 0, seg] = sampen
        results[ch, 1, seg] = apen
        results[ch, 2, seg] = spen

os.makedirs('result', exist_ok=True)
np.save('result/5_24.npy', results)
