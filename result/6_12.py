
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from scipy.integrate import simpson

# Load data
data = np.load('data/12_original.npy')

# Define sampling frequency (assuming 250Hz if not specified)
fs = 250  

# Function to compute band power
def bandpower(data, sf, band):
    band = np.asarray(band)
    low, high = band
    
    # Compute modified periodogram (Welch)
    freqs, psd = signal.welch(data, sf, nperseg=1024)
    
    # Find intersecting values in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    
    # Integral approximation of the spectrum using Simpson's rule
    bp = simpson(psd[idx_band], freqs[idx_band])
    
    return bp

# Compute Alpha and Beta power for each channel
alpha_power = np.array([bandpower(ch, fs, [8, 12]) for ch in data])
beta_power = np.array([bandpower(ch, fs, [13, 30]) for ch in data])

# Calculate cross-frequency correlation
correlation = pearsonr(alpha_power, beta_power)[0]
print(f"Cross-frequency correlation (Alpha-Beta): {correlation}")

# Prepare result array (7,1)
result = np.zeros((7, 1))
result[0, 0] = correlation  # Store correlation in first element

# Save results
np.save('result/6_12.npy', result)
