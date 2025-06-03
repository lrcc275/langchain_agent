
import numpy as np
from scipy.fft import fft

# Load the data from the .npy file
data = np.load('data/86_original.npy')

# Perform FFT on the data
fft_data = fft(data, axis=1)

# Calculate the amplitudes (absolute values)
amplitudes = np.abs(fft_data)

# Assuming the sampling rate is known (e.g., 250 Hz), we can calculate the frequency bins
sampling_rate = 250  # Adjust this if your sampling rate is different
n_samples = data.shape[1]
freqs = np.fft.fftfreq(n_samples, 1/sampling_rate)

# Find the index corresponding to 4Hz
target_freq = 4
idx = np.argmin(np.abs(freqs - target_freq))

# Extract the amplitudes at 4Hz
ssvep_amplitudes = amplitudes[:, idx]

# Reshape the result to (7, x)
result = ssvep_amplitudes.reshape(7, -1)

# Save the result to result/8_86.npy
np.save('result/8_86.npy', result)

# Print the result
print(result)
