import numpy as np
from sklearn.cluster import KMeans

# Load data
data = np.load('data/8_original.npy')

# Microstate analysis
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data.T)

# Get results
maps = kmeans.cluster_centers_

# Prepare and save results
result = np.zeros((7, 4))
n_maps = min(maps.shape[0], 4)
result[:n_maps, :n_maps] = maps[:n_maps, :n_maps]
np.save('result/11_8.npy', result)
