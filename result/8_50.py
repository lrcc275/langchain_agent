import numpy as np
from scipy.fft import fft

# Load the data
data = np.load('data/50_original.npy')

# Check data shape
if len(data.shape) == 1:
    data = data.reshape(1, -1)

# Parameters
sampling_rate = 250  # Hz
n_samples = data.shape[1]
n_channels = data.shape[0]

# Compute FFT for each channel
amplitudes = []
for channel_data in data:
    fft_result = fft(channel_data)
    frequencies = np.fft.fftfreq(n_samples, 1/sampling_rate)
    magnitudes = np.abs(fft_result) * 2 / n_samples
    
    # Find index for 4Hz
    target_freq = 4
    idx = np.argmin(np.abs(frequencies - target_freq))
    amplitudes.append(magnitudes[idx])

# Format result
if len(amplitudes) < 7:
    amplitudes.extend([0] * (7 - len(amplitudes)))
result = np.array(amplitudes[:7]).reshape(7, 1)

# Save the result
np.save('result/8_50.npy', result)