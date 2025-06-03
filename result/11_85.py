
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
data = np.load('data/85_original.npy')

# Assume data is channels x timepoints
if data.shape[0] > data.shape[1]:
    data = data.T

# Calculate GFP
gfp = np.std(data, axis=0)

# Find GFP peaks
peak_threshold = np.percentile(gfp, 90)
peak_indices = np.where(gfp >= peak_threshold)[0]
peak_data = data[:, peak_indices].T

# Perform k-means clustering
n_states = 4
kmeans = KMeans(n_clusters=n_states, random_state=42)
labels = kmeans.fit_predict(peak_data)
maps = kmeans.cluster_centers_

# Calculate silhouette score
silhouette = silhouette_score(peak_data, labels)

# Prepare results (7x4 array)
results = np.zeros((7,4))
results[:4,:] = maps.T[:4,:]
results[4,:] = np.mean(maps, axis=1)
results[5,:] = np.std(maps, axis=1)
results[6,:] = [silhouette]*4

# Save results
os.makedirs('result', exist_ok=True)
np.save('result/11_85.npy', results)
