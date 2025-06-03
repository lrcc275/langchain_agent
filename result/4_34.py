import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/34_original.npy')

# Parameters
fs = 250  # sample rate (Hz), typical for EEG
n_channels = data.shape[0]
freq_range = (8, 12)  # alpha band

# Initialize coherence matrix
coherence_matrix = np.zeros((n_channels, n_channels))

# Calculate coherence for each channel pair
for i in range(n_channels):
    for j in range(n_channels):
        if i <= j:  # avoid redundant calculations
            f, Cxy = signal.coherence(data[i], data[j], fs=fs)
            # Get indices for frequency range
            idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
            # Average coherence in the frequency range
            avg_coherence = np.mean(Cxy[idx])
            coherence_matrix[i,j] = avg_coherence
            coherence_matrix[j,i] = avg_coherence  # symmetric

# Print results
print("Coherence matrix (8-12Hz):")
print(coherence_matrix)

# Ensure result directory exists
os.makedirs('result', exist_ok=True)

# Save as (7,7) array
np.save('result/4_34.npy', coherence_matrix[:7,:7])
