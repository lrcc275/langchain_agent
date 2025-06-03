import numpy as np
from antropy import sample_entropy, app_entropy, spectral_entropy
import os

# Create result directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Load data
data = np.load('data/39_original.npy')

# Assuming sampling rate is 250 Hz (common for EEG), adjust if different
fs = 250
segment_length = 10 * fs  # 10 seconds in samples

# Calculate number of segments
n_segments = data.shape[1] // segment_length

# Initialize result array (7 channels, 3 entropy measures, n_segments)
result = np.zeros((7, 3, n_segments))

for seg in range(n_segments):
    start = seg * segment_length
    end = start + segment_length
    segment = data[:, start:end]
    
    for ch in range(7):  # Assuming 7 channels
        # Sample Entropy
        result[ch, 0, seg] = sample_entropy(segment[ch])
        
        # Approximate Entropy
        result[ch, 1, seg] = app_entropy(segment[ch])
        
        # Spectral Entropy
        result[ch, 2, seg] = spectral_entropy(segment[ch], sf=fs, method='fft')

# Save results
np.save('result/5_39.npy', result)