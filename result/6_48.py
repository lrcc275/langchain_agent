import numpy as np
from scipy import signal
import os

# Load the data
data = np.load('data/48_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250  

# Function to calculate band power
def bandpower(data, sf, band, window_sec=None, relative=False):
    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    nperseg = window_sec * sf if window_sec else None
    freqs, psd = signal.welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule
    bp = np.trapz(psd[idx_band], dx=freq_res)

    if relative:
        bp /= np.trapz(psd, dx=freq_res)
    return bp

# Calculate Alpha and Beta power for each channel and time window
alpha_power = np.array([bandpower(ch, fs, [8, 12]) for ch in data])
beta_power = np.array([bandpower(ch, fs, [13, 30]) for ch in data])

# Calculate correlation between Alpha and Beta power
correlation = np.corrcoef(alpha_power, beta_power)[0, 1]

# Reshape to (7,1) and save
result = np.array([[correlation]] * 7)
np.save('result/6_48.npy', result)
