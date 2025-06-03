import numpy as np
from scipy.fft import fft, fftfreq

data = np.load('data/69_original.npy')
fs = 250
target_freq = 4

amplitudes = []
for channel_data in data:
    n = len(channel_data)
    yf = fft(channel_data)
    xf = fftfreq(n, 1/fs)[:n//2]
    idx = np.argmin(np.abs(xf - target_freq))
    amplitude = 2/n * np.abs(yf[:n//2][idx])
    amplitudes.append(amplitude)

result = np.array(amplitudes).reshape(7, -1)
np.save('result/8_69.npy', result)