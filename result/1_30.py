import numpy as np

data = np.load('data/30_original.npy')

peak_to_peak = np.ptp(data, axis=1)
means = np.mean(data, axis=1)
variances = np.var(data, axis=1)

results = np.column_stack((peak_to_peak, means, variances))

print("Peak-to-peak values:", peak_to_peak)
print("Means:", means)
print("Variances:", variances)

np.save('result/1_30.npy', results)