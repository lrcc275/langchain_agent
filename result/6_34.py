import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# Load the data
data = np.load('data/34_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250

# Function to compute band power
def bandpower(data, fs, band):
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.mean(Pxx[idx])

# Define frequency bands
alpha_band = [8, 12]
beta_band = [13, 30]

# Calculate band powers for all channels
alpha_powers = np.array([bandpower(channel, fs, alpha_band) for channel in data])
beta_powers = np.array([bandpower(channel, fs, beta_band) for channel in data])

# Calculate cross-frequency correlation
correlation = pearsonr(alpha_powers, beta_powers)[0]

# Create (7,1) array by repeating the correlation value
result = np.full((7, 1), correlation)

# Save the result
np.save('result/6_34.npy', result)