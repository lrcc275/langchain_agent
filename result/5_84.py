import numpy as np
from scipy.stats import entropy
from antropy import sample_entropy, app_entropy, spectral_entropy
import os

# Load the data
data = np.load('data/84_original.npy')

# Parameters
fs = 250  # sampling frequency (Hz)
segment_length = 10 * fs  # 10 seconds in samples
n_segments = data.shape[1] // segment_length
n_channels = data.shape[0]

# Initialize result arrays
sample_entropies = np.zeros((n_channels, n_segments))
approx_entropies = np.zeros((n_channels, n_segments))
spectral_entropies = np.zeros((n_channels, n_segments))

# Calculate entropies for each channel and segment
for chan in range(n_channels):
    for seg in range(n_segments):
        start = seg * segment_length
        end = start + segment_length
        segment = data[chan, start:end]
        
        # Sample entropy
        sample_entropies[chan, seg] = sample_entropy(segment, order=2)
        
        # Approximate entropy
        approx_entropies[chan, seg] = app_entropy(segment, order=2)
        
        # Spectral entropy
        spectral_entropies[chan, seg] = spectral_entropy(segment, sf=fs, method='fft')

# Combine results into (7, 3, x) array where x is n_segments
result = np.stack([sample_entropies, approx_entropies, spectral_entropies], axis=1)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/5_84.npy', result)