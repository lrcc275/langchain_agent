import numpy as np
from antropy import sample_entropy, app_entropy, spectral_entropy

# Load data
data = np.load('data/63_original.npy')
fs = 250  # Assuming sampling rate is 250Hz
segment_length = 10 * fs  # 10 seconds of data

# Initialize result array (7 channels, 3 entropy measures, x segments)
num_channels = data.shape[0]
num_segments = data.shape[1] // segment_length
result = np.zeros((num_channels, 3, num_segments))

# Calculate entropies for each segment
for seg in range(num_segments):
    start = seg * segment_length
    end = start + segment_length
    for ch in range(num_channels):
        segment = data[ch, start:end]
        
        # Sample entropy
        sampen = sample_entropy(segment)
        # Approximate entropy
        apen = app_entropy(segment)
        # Spectral entropy
        spen = spectral_entropy(segment, sf=fs)
        
        result[ch, 0, seg] = sampen
        result[ch, 1, seg] = apen
        result[ch, 2, seg] = spen

# Save results
np.save('result/5_63.npy', result)