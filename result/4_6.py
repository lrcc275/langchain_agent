import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/6_original.npy')

# Parameters
fs = 1000  # Assuming sampling rate is 1000Hz
n_channels = data.shape[0]
freq_range = (8, 12)

# Initialize coherence matrix
coherence_matrix = np.zeros((n_channels, n_channels))

# Calculate coherence for each channel pair
for i in range(n_channels):
    for j in range(n_channels):
        if i <= j:  # Avoid redundant calculations
            f, Cxy = signal.coherence(data[i], data[j], fs=fs)
            # Get indices for the frequency range
            idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
            # Average coherence in the frequency range
            coherence_matrix[i, j] = np.mean(Cxy[idx])
            coherence_matrix[j, i] = coherence_matrix[i, j]

# Ensure result directory exists
os.makedirs('result', exist_ok=True)

# Save the result
np.save('result/4_6.npy', coherence_matrix)
