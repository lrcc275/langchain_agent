
import numpy as np
from sklearn.cluster import KMeans

# Load the data
data = np.load('data/80_original.npy')

# Basic microstate analysis using KMeans clustering
n_states = 4
kmeans = KMeans(n_clusters=n_states, n_init=10, max_iter=1000, tol=1e-6)
kmeans.fit(data.T)

# Get microstate maps
maps = kmeans.cluster_centers_.T

# Print results
print("Microstate maps:")
print(maps)

# Prepare and save results
result = maps[:7, :4]
np.save('result/11_80.npy', result)
