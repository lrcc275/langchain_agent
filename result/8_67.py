import numpy as np
from scipy.fft import fft

# Load the data
data = np.load('data/67_original.npy')

# Parameters
sampling_rate = 250  # Assuming typical EEG sampling rate of 250Hz
n_samples = data.shape[-1]  # Assuming time is the last dimension
frequencies = np.fft.fftfreq(n_samples, 1/sampling_rate)
target_freq = 4

# Initialize array to store amplitudes
if len(data.shape) == 1:
    # Single channel
    fft_values = fft(data)
    idx = np.argmin(np.abs(frequencies - target_freq))
    amplitude = np.abs(fft_values[idx])
    amplitudes = np.array([amplitude])
else:
    # Multi-channel (assuming channels first)
    amplitudes = []
    for channel in data:
        fft_values = fft(channel)
        idx = np.argmin(np.abs(frequencies - target_freq))
        amplitudes.append(np.abs(fft_values[idx]))
    amplitudes = np.array(amplitudes)

# Reshape to (7, x) format
n_channels = len(amplitudes)
x = int(np.ceil(n_channels / 7))
result = np.pad(amplitudes, (0, 7*x - n_channels), 'constant').reshape(7, x)

# Save the result
np.save('result/8_67.npy', result)