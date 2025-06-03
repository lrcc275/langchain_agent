import numpy as np
from scipy.fft import fft
import os

data = np.load('data/76_original.npy')
fs = 250
n = len(data)
freqs = np.fft.fftfreq(n, 1/fs)
fft_vals = fft(data)
magnitude = np.abs(fft_vals)
target_freq = 4
idx = np.argmin(np.abs(freqs - target_freq))
ssvep_4hz = magnitude[idx]
result = ssvep_4hz.reshape(7, -1)
os.makedirs('result', exist_ok=True)
np.save('result/8_76.npy', result)