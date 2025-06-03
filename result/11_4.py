import numpy as np
from sklearn.cluster import KMeans

# Load data
data = np.load('data/4_original.npy')

# Basic microstate analysis
# 1. Normalize data
data_normalized = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

# 2. Compute GFP
gfp = np.std(data_normalized, axis=0)

# 3. Find GFP peaks
n_peaks = min(1000, data.shape[1])
peak_indices = np.argsort(gfp)[-n_peaks:]
peaks = data_normalized[:, peak_indices].T

# 4. Cluster into microstates
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(peaks)
microstates = kmeans.cluster_centers_

# 5. Prepare and save result
result = microstates[:7, :] if microstates.shape[0] >= 7 else np.pad(microstates, ((0,7-microstates.shape[0]),(0,0)))
np.save('result/11_4.npy', result)