import numpy as np
from sklearn.cluster import KMeans

# Load the data
data = np.load('data/96_original.npy')

# Prepare data for clustering
data_for_clustering = data.T

# Microstate analysis
n_states = 4
kmeans = KMeans(n_clusters=n_states, random_state=42)
labels = kmeans.fit_predict(data_for_clustering)

# Get microstate maps
maps = kmeans.cluster_centers_

# Prepare and save results
result = np.zeros((7, 4))
result[:, :] = maps.T  # Transpose to (7,4)
np.save('result/11_96.npy', result)