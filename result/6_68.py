import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# Load the EEG data
data = np.load('data/68_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250  

# Function to extract band power
def get_band_power(data, band, fs):
    b, a = signal.butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='bandpass')
    filtered = signal.filtfilt(b, a, data)
    power = np.abs(signal.hilbert(filtered))**2
    return power

# Calculate Alpha and Beta power
alpha_power = get_band_power(data, [8, 12], fs)
beta_power = get_band_power(data, [13, 30], fs)

# Compute cross-frequency correlation
correlation, _ = pearsonr(alpha_power.flatten(), beta_power.flatten())
print(f"Cross-frequency correlation between Alpha and Beta bands: {correlation}")

# Create (7,1) array by repeating the correlation value
result = np.full((7, 1), correlation)

# Save the result
np.save('result/6_68.npy', result)