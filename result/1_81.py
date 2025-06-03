import numpy as np

data = np.load('data/81_original.npy')

peak_to_peak = np.ptp(data, axis=1)
means = np.mean(data, axis=1)
variances = np.var(data, axis=1)

results = np.column_stack((peak_to_peak, means, variances))

print("Channel Statistics:")
print("Peak-to-Peak\tMean\t\tVariance")
for i, (pp, mean, var) in enumerate(results):
    print(f"Channel {i+1}:\t{pp:.4f}\t{mean:.4f}\t{var:.4f}")

np.save('result/1_81.npy', results)
