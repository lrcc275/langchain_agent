
import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/40_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250

# Compute Alpha (8-12Hz) and Beta (13-30Hz) band power
def bandpower(data, fs, low, high):
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024)
    idx = np.logical_and(f >= low, f <= high)
    return np.mean(Pxx[idx])

alpha_power = np.array([bandpower(channel, fs, 8, 12) for channel in data])
beta_power = np.array([bandpower(channel, fs, 13, 30) for channel in data])

# Calculate cross-frequency correlation
correlation = np.corrcoef(alpha_power, beta_power)[0, 1]

# Create (7,1) array by repeating the correlation value
result = np.full((7, 1), correlation)

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/6_40.npy', result)
