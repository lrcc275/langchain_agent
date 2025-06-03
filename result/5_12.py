
import numpy as np
from antropy import sample_entropy, app_entropy, spectral_entropy
import os

# Load data
data = np.load('data/12_original.npy')

# Assuming sampling rate is 250 Hz (common for EEG), adjust if different
fs = 250
segment_length = 10 * fs  # 10 seconds of data

# Calculate number of segments
n_segments = data.shape[1] // segment_length

# Initialize result array (7 channels, 3 entropy measures, n_segments)
results = np.zeros((7, 3, n_segments))

for seg in range(n_segments):
    start = seg * segment_length
    end = start + segment_length
    segment = data[:, start:end]
    
    for ch in range(7):  # Assuming 7 channels
        # Calculate entropies
        samp_ent = sample_entropy(segment[ch])
        app_ent = app_entropy(segment[ch])
        spec_ent = spectral_entropy(segment[ch], sf=fs, method='welch')
        
        # Store results
        results[ch, 0, seg] = samp_ent
        results[ch, 1, seg] = app_ent
        results[ch, 2, seg] = spec_ent

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/5_12.npy', results)
