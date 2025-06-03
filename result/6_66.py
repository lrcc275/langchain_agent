import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# Load the EEG data
data = np.load('data/66_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250

# Define frequency bands
alpha_band = (8, 12)
beta_band = (13, 30)

# Function to extract band power
def get_band_power(data, band, fs):
    b, a = signal.butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='bandpass')
    filtered = signal.filtfilt(b, a, data)
    power = np.abs(signal.hilbert(filtered))**2
    return power

# Calculate power for each band
alpha_power = get_band_power(data, alpha_band, fs)
beta_power = get_band_power(data, beta_band, fs)

# Calculate cross-frequency correlation
correlation = np.zeros((7, 1))
for i in range(7):
    r, _ = pearsonr(alpha_power[i], beta_power[i])
    correlation[i] = r

# Save results
np.save('result/6_66.npy', correlation)