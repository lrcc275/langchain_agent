import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/23_original.npy')
n_channels = data.shape[0]

# Parameters
fs = 250  # sample rate (Hz), adjust if different
freq_range = (8, 12)  # alpha band
nperseg = 256  # segment length for coherence calculation

# Initialize coherence matrix
coh_matrix = np.zeros((n_channels, n_channels))

# Calculate coherence for all channel pairs
for i in range(n_channels):
    for j in range(i, n_channels):
        if i == j:
            coh_matrix[i,j] = 1.0  # coherence with itself is 1
        else:
            f, Cxy = signal.coherence(data[i], data[j], fs=fs, nperseg=nperseg)
            # Get average coherence in freq range
            idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
            coh_matrix[i,j] = coh_matrix[j,i] = np.mean(Cxy[idx])

# Print results
print("Coherence matrix (8-12Hz):")
print(coh_matrix)

# Ensure result directory exists
os.makedirs('result', exist_ok=True)

# Save as (7,7) array
np.save('result/4_23.npy', coh_matrix.reshape(7,7))
