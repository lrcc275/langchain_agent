import numpy as np
from sklearn.cluster import KMeans

# Load data
data = np.load('data/5_original.npy')

# Basic EEG microstate analysis with 4 clusters
n_clusters = 4

# Ensure data is in shape (n_samples, n_channels)
if data.shape[0] < data.shape[1]:
    data = data.T

# Normalize data
data_normalized = data / np.std(data, axis=0, keepdims=True)

# Perform k-means clustering
kmeans = KMeans(n_clusters=n_clusters, n_init=10)
kmeans.fit(data_normalized)

# Get cluster centers (microstate maps)
maps = kmeans.cluster_centers_.T

# Calculate GFP statistics
gfp = np.std(data_normalized, axis=1)
mean_gfp = np.mean(gfp)
max_gfp = np.max(gfp)
min_gfp = np.min(gfp)

# Prepare result array (7,4)
result = np.zeros((7, 4))
n_channels = min(maps.shape[0], 7)
result[:n_channels, :4] = maps[:n_channels, :]
result[n_channels:, 0] = mean_gfp
result[n_channels:, 1] = max_gfp
result[n_channels:, 2] = min_gfp

# Save results
np.save('result/11_5.npy', result)