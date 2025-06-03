import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/92_original.npy')

# Parameters
fs = 250  # sample rate (Hz), typical for EEG
n_channels = data.shape[0]
freq_range = (8, 12)  # Alpha band

# Compute coherence for all channel pairs
coherence_matrix = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(n_channels):
        if i <= j:  # Avoid redundant calculations
            f, Cxy = signal.coherence(data[i], data[j], fs=fs)
            # Get average coherence in the frequency range
            idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
            avg_coherence = np.mean(Cxy[idx])
            coherence_matrix[i,j] = avg_coherence
            coherence_matrix[j,i] = avg_coherence  # Symmetric matrix

# Print results
print("Coherence matrix (8-12Hz):")
print(coherence_matrix)

# Ensure result directory exists
os.makedirs('result', exist_ok=True)

# Save as (7,7) array
np.save('result/4_92.npy', coherence_matrix)
