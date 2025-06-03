import numpy as np
from scipy import signal

# Load the data
data = np.load('data/91_original.npy')

# Define sampling frequency
fs = 250  

# Calculate Alpha band (8-12Hz)
b_alpha, a_alpha = signal.butter(4, [8/(fs/2), 12/(fs/2)], btype='band')
alpha = signal.filtfilt(b_alpha, a_alpha, data)

# Calculate Beta band (13-30Hz)
b_beta, a_beta = signal.butter(4, [13/(fs/2), 30/(fs/2)], btype='band')
beta = signal.filtfilt(b_beta, a_beta, data)

# Calculate cross-frequency correlation
correlations = np.array([np.corrcoef(alpha[i], beta[i])[0,1] for i in range(7)]).reshape(7,1)

# Save results
np.save('result/6_91.npy', correlations)