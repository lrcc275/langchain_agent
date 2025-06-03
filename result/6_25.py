import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# Load the data
data = np.load('data/25_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250

# Function to extract frequency band
def get_band(data, low, high, fs):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# Extract Alpha and Beta bands
alpha = get_band(data, 8, 12, fs)
beta = get_band(data, 13, 30, fs)

# Calculate cross-frequency correlation
correlations = []
for i in range(7):  # Assuming 7 channels
    corr, _ = pearsonr(alpha[i], beta[i])
    correlations.append(corr)

# Convert to (7,1) array
result = np.array(correlations).reshape(7, 1)

# Save results
np.save('result/6_25.npy', result)