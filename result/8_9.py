import numpy as np
from scipy.fft import fft

# Load the original data
data = np.load('data/9_original.npy')

# Parameters (assuming typical values - you may need to adjust)
fs = 250  # sampling frequency in Hz
n_samples = data.shape[0] if len(data.shape) == 1 else data.shape[1]
freqs = np.fft.fftfreq(n_samples, 1/fs)

# Function to get SSVEP amplitude at target frequency
def get_ssvep_amplitude(signal, target_freq=4):
    fft_vals = fft(signal)
    magnitude = np.abs(fft_vals)
    idx = np.argmin(np.abs(freqs - target_freq))
    return magnitude[idx]

# Process data based on its shape
if len(data.shape) == 1:  # Single channel
    ssvep_amplitude = get_ssvep_amplitude(data)
    print(f"SSVEP amplitude at 4Hz: {ssvep_amplitude}")
    result = np.full((7, 1), ssvep_amplitude)  # Create (7,1) array
else:  # Multi-channel (assuming channels are in rows)
    ssvep_amplitudes = np.array([get_ssvep_amplitude(channel) for channel in data])
    print(f"SSVEP amplitudes at 4Hz: {ssvep_amplitudes}")
    # Reshape to have 7 rows (pad if necessary)
    n_channels = ssvep_amplitudes.shape[0]
    result = ssvep_amplitudes.reshape(-1, 1)  # Make column vector
    if n_channels < 7:
        result = np.pad(result, ((0, 7-n_channels), (0, 0)), 'constant')
    elif n_channels > 7:
        result = result[:7]

# Save result
np.save('result/8_9.npy', result)