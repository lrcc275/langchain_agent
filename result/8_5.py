import numpy as np
from scipy.fft import fft, fftfreq

# Load the original data
data = np.load('data/5_original.npy')

# Parameters
fs = 250  # Sampling rate
n = data.shape[1]
freqs = fftfreq(n, 1/fs)

# Initialize array to store SSVEP amplitudes
ssvep_amplitudes = np.zeros(7)

# Process each channel
for i in range(7):
    fft_vals = fft(data[i])
    magnitude = np.abs(fft_vals) * (2/n)
    target_freq = 4
    idx = np.argmin(np.abs(freqs - target_freq))
    ssvep_amplitudes[i] = magnitude[idx]

# Reshape to (7,1)
result = ssvep_amplitudes.reshape(7, 1)

# Save the result
np.save('result/8_5.npy', result)