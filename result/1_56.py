import numpy as np

data = np.load('data/56_original.npy')

peak_to_peak = np.ptp(data, axis=1)
means = np.mean(data, axis=1)
variances = np.var(data, axis=1)

results = np.column_stack((peak_to_peak, means, variances))

np.save('result/1_56.npy', results)