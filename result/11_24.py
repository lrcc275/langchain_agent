import numpy as np
from sklearn.cluster import KMeans

data = np.load('data/24_original.npy')
data_normalized = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_normalized.T)
maps = kmeans.cluster_centers_
result = maps.flatten()[:28].reshape(7, 4)
np.save('result/11_24.npy', result)