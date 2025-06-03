import numpy as np
from scipy import signal, stats

# Load data
data = np.load('data/52_original.npy')

# Function to calculate band power
def bandpower(data, sf, band):
    band = np.asarray(band)
    freqs, psd = signal.welch(data, sf, nperseg=4*sf)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bp = np.sum(psd[idx_band]) * freq_res
    return bp

# Calculate Alpha and Beta power for each channel
sf = 250  # sampling frequency
alpha_power = np.array([bandpower(ch, sf, [8, 12]) for ch in data])
beta_power = np.array([bandpower(ch, sf, [13, 30]) for ch in data])

# Calculate cross-frequency correlation
correlation = stats.pearsonr(alpha_power, beta_power)[0]
print(f"Cross-frequency correlation between Alpha and Beta bands: {correlation}")

# Create (7,1) array by repeating the correlation value
result = np.full((7, 1), correlation)
np.save('result/6_52.npy', result)