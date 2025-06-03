import numpy as np
from scipy.cluster.vq import kmeans2

# Load the data
data = np.load('data/54_original.npy')

# Calculate Global Field Power (GFP)
gfp = np.std(data, axis=0)

# Find GFP peaks (top 10% values)
threshold = np.percentile(gfp, 90)
peak_indices = np.where(gfp > threshold)[0]
peak_data = data[:, peak_indices].T

# Perform k-means clustering for microstates
n_microstates = 4
centroids, labels = kmeans2(peak_data, n_microstates, minit='++')

# Reshape centroids to (7 channels x 4 microstates)
microstate_maps = centroids.T

# Save results
np.save('result/11_54.npy', microstate_maps)
