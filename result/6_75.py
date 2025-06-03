import numpy as np
from scipy import signal

# Load the data
data = np.load('data/75_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250

# Compute Alpha band (8-12Hz)
alpha_band = [8, 12]
b_alpha, a_alpha = signal.butter(4, [alpha_band[0]/(fs/2), alpha_band[1]/(fs/2)], btype='bandpass')
alpha_data = signal.filtfilt(b_alpha, a_alpha, data)

# Compute Beta band (13-30Hz)
beta_band = [13, 30]
b_beta, a_beta = signal.butter(4, [beta_band[0]/(fs/2), beta_band[1]/(fs/2)], btype='bandpass')
beta_data = signal.filtfilt(b_beta, a_beta, data)

# Compute cross-frequency correlation
correlation = np.corrcoef(alpha_data.flatten(), beta_data.flatten())[0, 1]

# Print the result
print(f"Cross-frequency correlation between Alpha and Beta bands: {correlation}")

# Create (7,1) array by repeating the correlation value
result = np.full((7, 1), correlation)
np.save('result/6_75.npy', result)