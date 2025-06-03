import numpy as np
from scipy import signal
import os

def calculate_cfc(data, fs, low_freq_band, high_freq_band):
    b_alpha, a_alpha = signal.butter(4, [low_freq_band[0]/(fs/2), low_freq_band[1]/(fs/2)], btype='bandpass')
    alpha = signal.filtfilt(b_alpha, a_alpha, data)
    
    b_beta, a_beta = signal.butter(4, [high_freq_band[0]/(fs/2), high_freq_band[1]/(fs/2)], btype='bandpass')
    beta = signal.filtfilt(b_beta, a_beta, data)
    
    alpha_phase = np.angle(signal.hilbert(alpha))
    beta_amp = np.abs(signal.hilbert(beta))
    
    bins = np.linspace(-np.pi, np.pi, 19)
    bin_indices = np.digitize(alpha_phase, bins)
    mean_amp = np.array([beta_amp[bin_indices == i].mean() for i in range(1, len(bins))])
    mean_amp[np.isnan(mean_amp)] = 0
    mvl = np.abs(np.sum(mean_amp * np.exp(1j * bins[:-1]))) / np.sum(mean_amp)
    
    return mvl

data = np.load('data/55_original.npy')
fs = 250
results = np.zeros((7, 1))
for i in range(7):
    results[i] = calculate_cfc(data[i], fs, [8, 12], [13, 30])

print("Cross-frequency coupling results (Alpha-Beta):")
print(results)

os.makedirs('result', exist_ok=True)
np.save('result/6_55.npy', results)