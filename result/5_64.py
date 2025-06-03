import numpy as np
from antropy import sample_entropy, app_entropy, spectral_entropy
import os

data = np.load('data/64_original.npy')
fs = 1000
segment_length = 10 * fs
n_segments = data.shape[1] // segment_length
results = np.zeros((7, 3, n_segments))

for seg in range(n_segments):
    start = seg * segment_length
    end = (seg + 1) * segment_length
    segment = data[:, start:end]
    
    for ch in range(7):
        sampen = sample_entropy(segment[ch])
        apen = app_entropy(segment[ch])
        spen = spectral_entropy(segment[ch], sf=fs)
        
        results[ch, 0, seg] = sampen
        results[ch, 1, seg] = apen
        results[ch, 2, seg] = spen

print("Entropy results:")
print(results)

os.makedirs('result', exist_ok=True)
np.save('result/5_64.npy', results)