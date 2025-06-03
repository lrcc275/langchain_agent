import numpy as np
from scipy.fft import fft

# Load data
data = np.load('data/39_original.npy')

# Parameters
fs = 250  # Sample rate (Hz)
target_freq = 4  # Target SSVEP frequency (Hz)
n_samples = data.shape[1]  # Number of samples per channel

# Compute FFT and extract 4Hz amplitude
fft_values = np.abs(fft(data, axis=1))
freqs = np.fft.fftfreq(n_samples, 1/fs)
target_bin = np.argmin(np.abs(freqs - target_freq))

# Get amplitudes at 4Hz for all channels
ssvep_amplitudes = fft_values[:, target_bin]

# Reshape to (7, x) where x is automatically determined
result = ssvep_amplitudes.reshape(7, -1)

# Print results
print("SSVEP amplitudes at 4Hz:")
print(result)

# Save results
np.save('result/8_39.npy', result)