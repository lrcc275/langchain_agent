import numpy as np

data = np.load('data/28_original.npy')

peak_to_peak = np.ptp(data, axis=1)
mean = np.mean(data, axis=1)
variance = np.var(data, axis=1)

result = np.column_stack((peak_to_peak, mean, variance))

print("Peak-to-peak, Mean, Variance for each channel:")
print(result)

np.save('result/1_28.npy', result)