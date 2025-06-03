import numpy as np
from scipy.stats import entropy
from antropy import sample_entropy, spectral_entropy, app_entropy
import os

# Load the data
data = np.load('data/3_original.npy')
fs = 1000  # Assuming sampling rate is 1000Hz
window_size = 10 * fs  # 10 seconds window

# Calculate number of windows
n_windows = data.shape[1] // window_size

# Initialize result array (7 channels, 3 entropy measures, n_windows)
result = np.zeros((7, 3, n_windows))

# Calculate entropies for each window and channel
for win in range(n_windows):
    start = win * window_size
    end = (win + 1) * window_size
    for ch in range(7):  # Assuming 7 channels
        segment = data[ch, start:end]
        
        # Sample Entropy
        sampen = sample_entropy(segment, order=2)
        
        # Approximate Entropy
        apen = app_entropy(segment, order=2)
        
        # Spectral Entropy
        spen = spectral_entropy(segment, sf=fs, method='welch', normalize=True)
        
        result[ch, 0, win] = sampen
        result[ch, 1, win] = apen
        result[ch, 2, win] = spen

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/5_3.npy', result)