import numpy as np
from scipy import signal
from scipy.stats import pearsonr

data = np.load('data/82_original.npy')
fs = 250

# Alpha band
f_alpha = [8, 12]
b_alpha, a_alpha = signal.butter(4, [f_alpha[0]/(fs/2), f_alpha[1]/(fs/2)], 'bandpass')
alpha_data = signal.filtfilt(b_alpha, a_alpha, data)

# Beta band
f_beta = [13, 30]
b_beta, a_beta = signal.butter(4, [f_beta[0]/(fs/2), f_beta[1]/(fs/2)], 'bandpass')
beta_data = signal.filtfilt(b_beta, a_beta, data)

# Calculate correlations
correlations = []
for i in range(7):
    corr, _ = pearsonr(alpha_data[i], beta_data[i])
    correlations.append(corr)

result = np.array(correlations).reshape(7, 1)
np.save('result/6_82.npy', result)