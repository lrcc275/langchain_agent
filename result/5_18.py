
import numpy as np
from nolds import sampen
from antropy import app_entropy, spectral_entropy
import os

# Load data
data = np.load('data/18_original.npy')
fs = 250  # assuming sampling rate is 250Hz
segment_length = 10 * fs  # 10 seconds of data

# Calculate number of segments
n_segments = data.shape[1] // segment_length

# Initialize result array (7 channels, 3 entropy measures, n_segments)
results = np.zeros((7, 3, n_segments))

# Process each segment
for seg in range(n_segments):
    start = seg * segment_length
    end = start + segment_length
    segment = data[:, start:end]
    
    for ch in range(7):  # 7 channels
        channel_data = segment[ch]
        
        # Sample entropy
        samp_en = sampen(channel_data)
        
        # Approximate entropy
        app_en = app_entropy(channel_data, order=2)
        
        # Spectral entropy
        spec_en = spectral_entropy(channel_data, sf=fs, method='welch')
        
        results[ch, 0, seg] = samp_en
        results[ch, 1, seg] = app_en
        results[ch, 2, seg] = spec_en

# Print results
print("Entropy measures for each channel and segment:")
print(results)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/5_18.npy', results)
