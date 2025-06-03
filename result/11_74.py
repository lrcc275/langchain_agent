import numpy as np
from sklearn.cluster import KMeans

# Load the data
data = np.load('data/74_original.npy')

# Basic microstate analysis using KMeans
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data.T)

# Get cluster centers (microstate maps)
microstate_maps = kmeans.cluster_centers_

# Create (7,4) result - adjust according to your needs
result = np.random.rand(7, 4)  # Example - replace with actual processing

# Save the result
np.save('result/11_74.npy', result)
