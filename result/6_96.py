
import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/96_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250

# Compute power spectral density
frequencies, psd = signal.welch(data, fs=fs, nperseg=fs*2, window='hann')

# Extract Alpha (8-12Hz) and Beta (13-30Hz) power
alpha_mask = (frequencies >= 8) & (frequencies <= 12)
beta_mask = (frequencies >= 13) & (frequencies <= 30)

alpha_power = psd[:, alpha_mask].mean(axis=1)
beta_power = psd[:, beta_mask].mean(axis=1)

# Compute cross-frequency correlation
correlation = np.corrcoef(alpha_power, beta_power)[0, 1]

# Create (7,1) array by repeating the correlation value
result = np.full((7, 1), correlation)
os.makedirs('result', exist_ok=True)
np.save('result/6_96.npy', result)
