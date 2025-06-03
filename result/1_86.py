import numpy as np

# Load the data
data = np.load('data/86_original.npy')

# Calculate statistics for each channel
peak_to_peak = np.ptp(data, axis=1)
mean = np.mean(data, axis=1)
variance = np.var(data, axis=1)

# Combine results into a (7,3) array
results = np.column_stack((peak_to_peak, mean, variance))

# Print the results
print("Peak-to-peak, Mean, Variance for each channel:")
print(results)

# Save the results
np.save('result/1_86.npy', results)
