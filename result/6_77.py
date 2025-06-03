
import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# Load the data
data = np.load('data/77_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250

# Function to extract frequency band
def extract_band(data, low, high, fs):
    nyquist = 0.5 * fs
    low_norm = low / nyquist
    high_norm = high / nyquist
    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
    return signal.filtfilt(b, a, data)

# Extract Alpha and Beta bands
alpha_band = extract_band(data, 8, 12, fs)
beta_band = extract_band(data, 13, 30, fs)

# Compute cross-frequency correlation
correlation = np.array([pearsonr(alpha_band[:, i], beta_band[:, i])[0] for i in range(7)]).reshape(7, 1)

# Save the result
np.save('result/6_77.npy', correlation)
