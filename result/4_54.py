import numpy as np
from scipy import signal
import os

# Load data
data = np.load('data/54_original.npy')

# Parameters
fs = 250  # sample rate (Hz), typical for EEG
n_channels = data.shape[0]
freq_range = (8, 12)  # Alpha band

# Calculate coherence for all channel pairs
coherence_matrix = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(n_channels):
        if i <= j:  # Avoid redundant calculations
            f, Cxy = signal.coherence(data[i], data[j], fs=fs)
            # Get average coherence in the frequency band
            mask = (f >= freq_range[0]) & (f <= freq_range[1])
            coherence_matrix[i,j] = np.mean(Cxy[mask])
            coherence_matrix[j,i] = coherence_matrix[i,j]

# Print results
print("Coherence matrix (8-12Hz):")
print(coherence_matrix)

# Ensure result directory exists
os.makedirs('result', exist_ok=True)

# Save as (7,7) array
np.save('result/4_54.npy', coherence_matrix)