
import numpy as np
from sklearn.cluster import KMeans

# Load data
data = np.load('data/67_original.npy')

# Transpose if necessary
if data.shape[0] < data.shape[1]:
    data = data.T

# Perform K-means clustering
n_states = 4
kmeans = KMeans(n_clusters=n_states, random_state=42)
kmeans.fit(data)

# Get microstate maps
maps = kmeans.cluster_centers_

# Ensure shape is (7,4)
if maps.shape[1] > 7:
    maps = maps[:, :7].T
elif maps.shape[0] > 7:
    maps = maps[:7, :]

if maps.shape != (7,4):
    maps = maps[:7, :4]

# Save results
np.save('result/11_67.npy', maps)
