import numpy as np
import scipy.signal as signal

# Load the data
data = np.load('data/86_original.npy')

# Parameters
fs = 1000  # Assuming sampling rate is 1000Hz
n_channels = data.shape[0]
freq_range = (8, 12)

# Compute coherence for all channel pairs
coherence_matrix = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(n_channels):
        if i == j:
            coherence_matrix[i, j] = 1.0  # coherence with itself is 1
        else:
            f, Cxy = signal.coherence(data[i], data[j], fs=fs)
            # Get the average coherence in the frequency range
            idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
            coherence_matrix[i, j] = np.mean(Cxy[idx])

# Save the result
np.save('result/4_86.npy', coherence_matrix)