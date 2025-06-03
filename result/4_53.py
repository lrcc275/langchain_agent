
import numpy as np
import scipy.signal as signal
from scipy.signal import coherence

# Load the data
data = np.load('data/53_original.npy')

# Compute coherence for all channel pairs (7 channels)
n_channels = data.shape[0]
coh_matrix = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(n_channels):
        f, Cxy = coherence(data[i], data[j], fs=250, nperseg=1024)
        # Select frequency band 8-12Hz
        band = (f >= 8) & (f <= 12)
        coh_matrix[i, j] = np.mean(Cxy[band])

# Print the result
print("Coherence matrix (8-12Hz):")
print(coh_matrix)

# Save the result
np.save('result/4_53.npy', coh_matrix)
