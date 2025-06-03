import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/15_original.npy')

# Parameters
fs = 1000  # Assuming a sampling rate of 1000 Hz
n_channels = data.shape[0]
freq_range = (8, 12)

# Initialize coherence matrix
coherence_matrix = np.zeros((n_channels, n_channels))

# Calculate coherence for each channel pair
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            f, Cxy = signal.coherence(data[i], data[j], fs=fs)
            idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
            coherence_matrix[i, j] = np.mean(Cxy[idx])

# Print the result
print(coherence_matrix)

# Save the result
os.makedirs('result', exist_ok=True)
np.save('result/4_15.npy', coherence_matrix.reshape(7, 7))
