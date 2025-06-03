import numpy as np
from scipy.stats import entropy
from antropy import sample_entropy, spectral_entropy
from scipy.signal import welch
import nolds

# Load the data
data = np.load('data/44_original.npy')
fs = 250  # Assuming sampling rate is 250Hz
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
        try:
            # Sample Entropy
            sampen = sample_entropy(segment[ch], order=2)
        except:
            sampen = np.nan
            
        try:
            # Approximate Entropy (using sample entropy from nolds as alternative)
            apen = nolds.sampen(segment[ch])
        except:
            apen = np.nan
            
        try:
            # Spectral Entropy
            nperseg = min(256, len(segment[ch])//2)  # Ensure reasonable segment length
            freqs, psd = welch(segment[ch], fs=fs, nperseg=nperseg)
            spen = spectral_entropy(psd, freqs, normalize=True)
        except:
            spen = np.nan
        
        results[ch, 0, seg] = sampen
        results[ch, 1, seg] = apen
        results[ch, 2, seg] = spen

# Save results
np.save('result/5_44.npy', results)