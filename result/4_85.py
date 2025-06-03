import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/85_original.npy')

# Parameters
fs = 250  # sampling frequency (Hz)
n_channels = data.shape[0]
freq_range = (8, 12)  # frequency range of interest (Hz)

# Compute coherence for all channel pairs
coherence_matrix = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(n_channels):
        if i <= j:  # avoid redundant calculations
            f, Cxy = signal.coherence(data[i], data[j], fs=fs)
            # Get average coherence in the frequency range
            idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
            avg_coherence = np.mean(Cxy[idx])
            coherence_matrix[i,j] = avg_coherence
            coherence_matrix[j,i] = avg_coherence  # matrix is symmetric

# Save the result
os.makedirs('result', exist_ok=True)
np.save('result/4_85.npy', coherence_matrix)