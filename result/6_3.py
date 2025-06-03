import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# Load the data
data = np.load('data/3_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250  

# Function to calculate band power
def bandpower(data, fs, band):
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024)
    idx = np.where((f >= band[0]) & (f <= band[1]))[0]
    return np.trapz(Pxx[idx], f[idx])

# Calculate Alpha and Beta power for each channel
alpha_power = np.array([bandpower(ch, fs, [8, 12]) for ch in data])
beta_power = np.array([bandpower(ch, fs, [13, 30]) for ch in data])

# Calculate cross-frequency correlation
correlation = pearsonr(alpha_power, beta_power)[0]
print(f"Cross-frequency correlation between Alpha and Beta bands: {correlation}")

# Create (7,1) array by repeating the correlation value
result = np.full((7, 1), correlation)
np.save('result/6_3.npy', result)