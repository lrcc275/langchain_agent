import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/36_original.npy')
n_channels = data.shape[0]

# Parameters
fs = 250  # sampling frequency (adjust if different)
freq_range = (8, 12)  # alpha band

# Initialize coherence matrix
coherence_matrix = np.zeros((n_channels, n_channels))

# Calculate coherence for all channel pairs
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            f, Cxy = signal.coherence(data[i], data[j], fs=fs)
            # Get average coherence in the specified frequency range
            idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
            coherence_matrix[i,j] = np.mean(Cxy[idx])

# Print results
print("Coherence matrix (8-12Hz):")
print(coherence_matrix)

# Ensure result directory exists
os.makedirs('result', exist_ok=True)

# Save as (7,7) array
np.save('result/4_36.npy', coherence_matrix.reshape(7,7))
