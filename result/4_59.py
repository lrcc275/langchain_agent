import numpy as np
import scipy.signal as signal
from pathlib import Path

# Load data
data = np.load('data/59_original.npy')
fs = 250  # Assuming sampling rate is 250Hz

# Compute coherence for all channel pairs (7 channels)
n_channels = data.shape[0]
coh_matrix = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(n_channels):
        f, Cxy = signal.coherence(data[i], data[j], fs=fs, nperseg=1024)
        # Get average coherence in 8-12Hz band
        band_mask = (f >= 8) & (f <= 12)
        coh_matrix[i, j] = np.mean(Cxy[band_mask])

# Save as (7,7) array
Path('result').mkdir(exist_ok=True)
np.save('result/4_59.npy', coh_matrix)