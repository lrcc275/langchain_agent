import numpy as np
from scipy import signal

data = np.load('data/4_original.npy')
fs = 250

# Alpha band
b_alpha, a_alpha = signal.butter(4, [8/(fs/2), 12/(fs/2)], btype='bandpass')
alpha = signal.filtfilt(b_alpha, a_alpha, data)

# Beta band
b_beta, a_beta = signal.butter(4, [13/(fs/2), 30/(fs/2)], btype='bandpass')
beta = signal.filtfilt(b_beta, a_beta, data)

# Correlation
correlation = np.corrcoef(alpha.flatten(), beta.flatten())[0, 1]
result = np.full((7, 1), correlation)
np.save('result/6_4.npy', result)