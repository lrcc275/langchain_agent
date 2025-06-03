import numpy as np
from antropy import sample_entropy, app_entropy, spectral_entropy
import os

# Load the original data
data = np.load('data/38_original.npy')

# Assuming sampling rate is 250Hz (common for EEG), adjust if different
sampling_rate = 250
segment_length = 10 * sampling_rate  # 10 seconds in samples

# Calculate number of segments
num_segments = data.shape[1] // segment_length

# Initialize result array (7 channels, 3 entropy measures, x segments)
results = np.zeros((7, 3, num_segments))

for seg in range(num_segments):
    start = seg * segment_length
    end = (seg + 1) * segment_length
    segment = data[:, start:end]
    
    for channel in range(7):  # Assuming 7 channels
        # Calculate entropy measures
        sampen = sample_entropy(segment[channel])
        apen = app_entropy(segment[channel])
        spen = spectral_entropy(segment[channel], sf=sampling_rate)
        
        # Store results
        results[channel, 0, seg] = sampen
        results[channel, 1, seg] = apen
        results[channel, 2, seg] = spen

# Create result directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Save results
np.save('result/5_38.npy', results)