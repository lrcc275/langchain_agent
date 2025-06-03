import numpy as np
from antropy import sample_entropy, app_entropy, spectral_entropy
import os

data = np.load('data/54_original.npy')
fs = 250
segment_length = 10 * fs

n_segments = data.shape[1] // segment_length
results = np.zeros((7, 3, n_segments))

for seg in range(n_segments):
    start = seg * segment_length
    end = start + segment_length
    segment = data[:, start:end]
    
    for ch in range(7):
        results[ch, 0, seg] = sample_entropy(segment[ch])
        results[ch, 1, seg] = app_entropy(segment[ch])
        results[ch, 2, seg] = spectral_entropy(segment[ch], sf=fs, method='welch')

print("Entropy results:")
print(results)

os.makedirs('result', exist_ok=True)
np.save('result/5_54.npy', results)