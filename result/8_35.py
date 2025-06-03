import numpy as np
from scipy.fft import fft

# Load the data
data = np.load('data/35_original.npy')

# Parameters
fs = 1000  # Assuming sampling rate is 1000Hz
n = len(data)
freq = np.fft.fftfreq(n, 1/fs)
target_freq = 4  # Target frequency in Hz
tolerance = 0.1  # Frequency tolerance in Hz

# Perform FFT
fft_values = fft(data)
magnitude = np.abs(fft_values)

# Find indices near 4Hz
idx = np.where((freq >= target_freq - tolerance) & (freq <= target_freq + tolerance))[0]

if len(idx) == 0:
    print("No frequency components found near 4Hz")
    result = np.zeros((7, 1))  # Default empty result
else:
    print(f"Found {len(idx)} frequency components near 4Hz")
    amplitudes = magnitude[idx]
    
    # Reshape to (7, x) format
    x = len(amplitudes) // 7
    if len(amplitudes) % 7 != 0:
        x += 1
    result = amplitudes[:7*x].reshape(7, -1)

# Print results
print("Final amplitudes at 4Hz (±0.1Hz):")
print(result)

# Save results
np.save('result/8_35.npy', result)