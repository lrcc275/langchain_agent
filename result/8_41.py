
import numpy as np
from scipy.fft import fft

# Load the original data (assuming shape is (7, n_samples))
data = np.load('data/41_original.npy')

# Check data shape and reshape if necessary
if len(data.shape) == 1:
    # If data is 1D, reshape to (7, n_samples)
    n_samples = len(data) // 7
    data = data.reshape(7, n_samples)

# Parameters
fs = 250  # Sampling rate (Hz)
target_freq = 4  # Target frequency (Hz)

# Initialize array to store amplitudes
amplitudes = np.zeros(7)

# Process each channel
for i in range(7):
    channel_data = data[i]
    n = len(channel_data)
    
    # Perform FFT
    fft_result = fft(channel_data)
    frequencies = np.fft.fftfreq(n, 1/fs)
    magnitude = np.abs(fft_result)
    
    # Find the index closest to 4Hz
    idx = np.argmin(np.abs(frequencies - target_freq))
    amplitudes[i] = magnitude[idx]

# Reshape to (7, 1) and save
result = amplitudes.reshape(7, 1)
np.save('result/8_41.npy', result)
