import numpy as np
from scipy.fft import fft

# Load data
data = np.load('data/73_original.npy')

# Parameters
fs = 250  # Assuming sampling rate is 250Hz
target_freq = 4  # Target SSVEP frequency
n_samples = data.shape[1]  # Number of time samples

# Compute FFT for each channel (7 channels)
fft_results = []
for channel in data:
    fft_data = fft(channel)
    freqs = np.fft.fftfreq(n_samples, 1/fs)
    idx = np.argmin(np.abs(freqs - target_freq))
    amplitude = np.abs(fft_data[idx])
    fft_results.append(amplitude)

# Convert to (7,1) array
result = np.array(fft_results).reshape(7, -1)

# Save results
np.save('result/8_73.npy', result)