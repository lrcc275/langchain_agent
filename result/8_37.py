import numpy as np
from scipy.fft import fft

# Load the data
data = np.load('data/37_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz (common for EEG)
target_freq = 4  # Target frequency in Hz
n_samples = data.shape[1]  # Assuming shape is (channels, timepoints)

# Initialize array to store results
results = []

# Process each channel
for channel_data in data:
    # Compute FFT
    fft_result = fft(channel_data)
    frequencies = np.fft.fftfreq(n_samples, 1/fs)
    magnitude = np.abs(fft_result)
    
    # Find the index corresponding to 4Hz
    target_idx = np.argmin(np.abs(frequencies - target_freq))
    ssvep_amplitude = magnitude[target_idx]
    results.append(ssvep_amplitude)

# Reshape to (7, x) format - assuming 7 channels
result = np.array(results).reshape(7, -1)

# Save the result
np.save('result/8_37.npy', result)