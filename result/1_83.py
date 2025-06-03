import numpy as np

# Load the data
data = np.load('data/83_original.npy')

# Compute metrics for each channel (assuming channels are rows)
metrics = np.zeros((7, 3))  # 7 channels, 3 metrics each
for i in range(7):
    channel_data = data[i]
    metrics[i, 0] = np.ptp(channel_data)  # Peak-to-peak
    metrics[i, 1] = np.mean(channel_data)  # Mean
    metrics[i, 2] = np.var(channel_data)   # Variance

# Print results
print("Channel metrics (peak-to-peak, mean, variance):")
print(metrics)

# Save results
np.save('result/1_83.npy', metrics)
