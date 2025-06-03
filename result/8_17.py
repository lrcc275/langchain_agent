
import numpy as np
from scipy.fft import fft

# Load the data
data = np.load('data/17_original.npy')

# Assuming data is (channels, timepoints)
if len(data.shape) == 1:
    data = data.reshape(1, -1)  # Make it 2D if it's 1D

# Parameters
sampling_rate = 250  # Typical EEG sampling rate
target_freq = 4  # SSVEP frequency of interest
n_points = data.shape[1]  # Number of timepoints

# Initialize array to store amplitudes
amplitudes = np.zeros(data.shape[0])  # One amplitude per channel

# Compute FFT and extract amplitude for each channel
for i in range(data.shape[0]):
    fft_data = fft(data[i])
    freqs = np.fft.fftfreq(n_points, 1/sampling_rate)
    idx = np.argmin(np.abs(freqs - target_freq))
    amplitudes[i] = np.abs(fft_data[idx])

# Print results
print(f"Amplitudes at {target_freq}Hz for each channel:")
print(amplitudes)

# Reshape to (7,x) format
n_channels = data.shape[0]
result = amplitudes.reshape(7, -1)  # Will work if n_channels is divisible by 7

# Save the result
np.save('result/8_17.npy', result)
