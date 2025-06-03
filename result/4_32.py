import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('./data/32_original.npy')

# Parameters
fs = 250  # sample rate (Hz), typical for EEG
nperseg = 256  # segment length for coherence calculation
freq_range = (8, 12)  # alpha band

# Calculate coherence for all channel pairs
n_channels = data.shape[0]
coherence_matrix = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(i, n_channels):
        if i == j:
            coherence_matrix[i, j] = 1.0  # coherence with itself is 1
        else:
            f, Cxy = signal.coherence(data[i], data[j], fs=fs, nperseg=nperseg)
            # Get average coherence in the frequency range
            idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
            coherence_matrix[i, j] = np.mean(Cxy[idx])
            coherence_matrix[j, i] = coherence_matrix[i, j]  # symmetric

# Save the result
os.makedirs('./result', exist_ok=True)
np.save('./result/4_32.npy', coherence_matrix)