
import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/50_original.npy')

# Parameters
fs = 1000  # Assuming sampling rate is 1000Hz
n_channels = data.shape[0]
freq_range = (8, 12)

# Compute coherence
coherence_matrix = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            f, Cxy = signal.coherence(data[i], data[j], fs=fs)
            idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
            coherence_matrix[i, j] = np.mean(Cxy[idx])

# Print results
print("Coherence matrix (8-12Hz):")
print(coherence_matrix)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/4_50.npy', coherence_matrix)
