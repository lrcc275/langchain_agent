import numpy as np
import mne
from sklearn.cluster import KMeans

# Load data
data = np.load('data/23_original.npy')

# Prepare data (transpose)
data = data.T

# Microstate analysis
n_states = 4
kmeans = KMeans(n_clusters=n_states, random_state=42)
kmeans.fit(data)

# Get and reshape maps
maps = kmeans.cluster_centers_
result = maps.reshape(7, 4)

# Save results
np.save('result/11_23.npy', result)