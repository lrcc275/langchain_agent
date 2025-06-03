import numpy as np
from scipy import signal
from scipy.stats import pearsonr

data = np.load('data/18_original.npy')
fs = 250

def get_band(data, low, high, fs):
    b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return signal.filtfilt(b, a, data)

alpha = get_band(data, 8, 12, fs)
beta = get_band(data, 13, 30, fs)

correlations = []
for i in range(7):
    corr, _ = pearsonr(alpha[i], beta[i])
    correlations.append(corr)

result = np.array(correlations).reshape(7,1)
np.save('result/6_18.npy', result)