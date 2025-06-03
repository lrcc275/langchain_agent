import numpy as np

# Load the data
data = np.load('data/80_original.npy')

# Compute statistics for each channel
peak_to_peak = np.ptp(data, axis=1)
means = np.mean(data, axis=1)
variances = np.var(data, axis=1)

# Combine results into (7,3) array
results = np.column_stack((peak_to_peak, means, variances))

# Print results
print("Channel Statistics:")
print("Peak-to-Peak\tMean\t\tVariance")
for i, (pp, mean, var) in enumerate(results):
    print(f"Channel {i+1}: {pp:.3f}\t{mean:.3f}\t{var:.3f}")

# Save results
np.save('result/1_80.npy', results)
