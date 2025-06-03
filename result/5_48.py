import numpy as np
from antropy import sample_entropy, app_entropy, spectral_entropy

# Load the original data
data = np.load('data/48_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
segment_length = 10 * fs  # 10 seconds in samples
n_segments = data.shape[1] // segment_length
n_channels = data.shape[0]

# Initialize result array (7 channels, 3 entropy measures, x segments)
results = np.zeros((7, 3, n_segments))

# Calculate entropies for each segment
for seg in range(n_segments):
    start = seg * segment_length
    end = start + segment_length
    for ch in range(7):  # Assuming 7 channels
        segment = data[ch, start:end]
        
        # Sample entropy
        sampen = sample_entropy(segment)
        
        # Approximate entropy
        apen = app_entropy(segment)
        
        # Spectral entropy
        spen = spectral_entropy(segment, sf=fs, method='fft')
        
        results[ch, 0, seg] = sampen
        results[ch, 1, seg] = apen
        results[ch, 2, seg] = spen

# Save results
np.save('result/5_48.npy', results)