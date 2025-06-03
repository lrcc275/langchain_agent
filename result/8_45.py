import numpy as np
from scipy.fft import fft

# Load data
data = np.load('data/45_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
target_freq = 4  # Hz
n_samples = len(data)

# Perform FFT
fft_result = fft(data)
frequencies = np.fft.fftfreq(n_samples, 1/fs)
magnitude = np.abs(fft_result)

# Find index for 4Hz
idx = np.argmin(np.abs(frequencies - target_freq))
amp_at_4hz = magnitude[idx]

# Calculate required padding to make divisible by 7
pad_size = (7 - (len(amp_at_4hz) % 7)) % 7
padded = np.pad(amp_at_4hz, (0, pad_size), 'constant')

# Reshape to (7,x) format
result = padded.reshape(7, -1)

# Save results
np.save('result/8_45.npy', result)