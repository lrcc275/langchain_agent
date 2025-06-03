import numpy as np
from scipy import signal

# Load data
data = np.load('data/28_original.npy')

# Parameters
fs = 250  # sampling frequency (Hz)
alpha_band = (8, 12)  # Alpha frequency range
beta_band = (13, 30)  # Beta frequency range

# Function to calculate band power
def bandpower(data, sf, band):
    freqs, psd = signal.welch(data, sf, nperseg=1024)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[idx_band])

# Calculate Alpha and Beta power for each channel
alpha_power = np.array([bandpower(ch, fs, alpha_band) for ch in data])
beta_power = np.array([bandpower(ch, fs, beta_band) for ch in data])

# Calculate cross-frequency correlation
correlation = np.corrcoef(alpha_power, beta_power)[0, 1]

# Print results
print(f"Cross-frequency correlation between Alpha and Beta bands: {correlation}")

# Create (7,1) array by repeating the correlation value and save
result = np.full((7, 1), correlation)
np.save('result/6_28.npy', result)