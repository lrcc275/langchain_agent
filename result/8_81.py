import numpy as np
from scipy.fft import fft

# Load the data
data = np.load('data/81_original.npy')

# Process the data
n_channels, n_samples = data.shape
sampling_rate = 250  # Adjust if needed
freqs = np.fft.fftfreq(n_samples, 1/sampling_rate)
idx = np.argmin(np.abs(freqs - 4))  # Find 4Hz index

amplitudes = []
for channel in data:
    fft_data = fft(channel)
    amplitudes.append(np.abs(fft_data[idx]))
amplitudes = np.array(amplitudes)

# Reshape and save
result = amplitudes.reshape(7, 1)
np.save('result/8_81.npy', result)
