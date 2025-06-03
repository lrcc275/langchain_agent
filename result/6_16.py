
import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# Load the data
data = np.load('data/16_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250

# Define Alpha and Beta bands
alpha_band = (8, 12)
beta_band = (13, 30)

# Function to extract band power
def get_band_power(data, band, fs):
    # Design bandpass filter
    b, a = signal.butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    # Apply filter
    filtered = signal.filtfilt(b, a, data)
    # Compute power (squared amplitude)
    power = np.abs(filtered)**2
    return power

# Compute Alpha and Beta power
alpha_power = get_band_power(data, alpha_band, fs)
beta_power = get_band_power(data, beta_band, fs)

# Compute cross-frequency correlation for each channel
correlation = np.array([pearsonr(alpha_power[i], beta_power[i])[0] for i in range(len(data))])

# Reshape to (7,1)
result = correlation.reshape(-1, 1)

# Save the result
np.save('result/6_16.npy', result)
