
import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# Load the data
data = np.load('data/51_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250  # Modify if your sampling frequency is different

# Define frequency bands
alpha_band = (8, 12)  # Alpha band
beta_band = (13, 30)  # Beta band

# Function to compute band power
def compute_band_power(data, band, fs):
    # Design bandpass filter
    b, a = signal.butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='bandpass')
    # Apply filter
    filtered = signal.filtfilt(b, a, data)
    # Compute power (squared amplitude)
    power = np.abs(filtered)**2
    return power

# Compute power in Alpha and Beta bands
alpha_power = compute_band_power(data, alpha_band, fs)
beta_power = compute_band_power(data, beta_band, fs)

# Compute cross-frequency correlation
correlation = np.zeros((7, 1))  # Initialize (7,1) array
for i in range(7):  # Assuming 7 channels (modify if different)
    corr, _ = pearsonr(alpha_power[i], beta_power[i])
    correlation[i] = corr

# Print the result
print("Cross-frequency correlation (Alpha-Beta):")
print(correlation)

# Save the result
np.save('result/6_51.npy', correlation)
