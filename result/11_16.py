import numpy as np
from sklearn.cluster import KMeans

# Load data
data = np.load('data/16_original.npy')

# Check data shape and transpose if needed
if data.shape[0] == 16:
    data = data.T

# Perform K-means clustering for microstates
n_states = 4
kmeans = KMeans(n_clusters=n_states, random_state=42)
kmeans.fit(data)

# Get microstate maps (cluster centers)
maps = kmeans.cluster_centers_

# Select first 7 channels for each state
selected_maps = maps[:7, :].T

# Ensure final shape is (7,4)
result = selected_maps[:7, :4]

# Save results
np.save('result/11_16.npy', result)