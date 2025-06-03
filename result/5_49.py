import numpy as np
from nolds import sampen
from antropy import spectral_entropy, app_entropy
import os

os.makedirs('result', exist_ok=True)

data = np.load('data/49_original.npy')
fs = 250
segment_length = 10 * fs
n_segments = data.shape[1] // segment_length
results = np.zeros((7, 3, n_segments))

for seg in range(n_segments):
    start = seg * segment_length
    end = (seg + 1) * segment_length
    segment = data[:, start:end]
    
    for ch in range(7):
        results[ch, 0, seg] = sampen(segment[ch])
        results[ch, 1, seg] = app_entropy(segment[ch])
        results[ch, 2, seg] = spectral_entropy(segment[ch], sf=fs, method='fft')

np.save('result/5_49.npy', results)