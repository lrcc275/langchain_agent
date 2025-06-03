import numpy as np
from scipy import signal
import os

data = np.load('data/48_original.npy')

fs = 250
n_channels = data.shape[0]
freq_range = (8, 12)
nperseg = 256

coh_matrix = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(n_channels):
        if i <= j:
            f, Cxy = signal.coherence(data[i], data[j], fs=fs, nperseg=nperseg)
            mask = (f >= freq_range[0]) & (f <= freq_range[1])
            coh_matrix[i,j] = coh_matrix[j,i] = np.mean(Cxy[mask])

print("Coherence matrix (8-12Hz):")
print(coh_matrix)

os.makedirs('result', exist_ok=True)
np.save('result/4_48.npy', coh_matrix)