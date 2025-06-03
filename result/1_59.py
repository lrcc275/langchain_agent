import numpy as np

# Load the data
data = np.load('data/59_original.npy')

# Calculate statistics for each channel
peak_to_peak = np.ptp(data, axis=1)
means = np.mean(data, axis=1)
variances = np.var(data, axis=1)

# Combine results into (7,3) array
results = np.column_stack((peak_to_peak, means, variances))

# Print results
print("Peak-to-peak values:", peak_to_peak)
print("Means:", means)
print("Variances:", variances)
print("Combined results array:")
print(results)

# Save results
np.save('result/1_59.npy', results)
