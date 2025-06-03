import numpy as np
import mne
from sklearn.cluster import KMeans

# Load data
data = np.load('data/20_original.npy')

# Calculate GFP and find peaks
gfp = np.std(data, axis=0)
peaks = np.where(gfp > np.percentile(gfp, 90))[0]
gfp_peaks = data[:, peaks].T

# Perform k-means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(gfp_peaks)

# Get microstate maps
maps = kmeans.cluster_centers_.T

# Prepare and save result
result = maps[:7, :4] if maps.shape[0] >= 7 and maps.shape[1] >= 4 else np.pad(maps, ((0, max(0,7-maps.shape[0])), (0, max(0,4-maps.shape[1]))), 'constant')
np.save('result/11_20.npy', result)