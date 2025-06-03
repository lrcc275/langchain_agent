import numpy as np
from sklearn.cluster import KMeans

data = np.load('data/68_original.npy').T
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data)
microstate_maps = kmeans.cluster_centers_
print("Microstate Maps:\n", microstate_maps)
np.save('result/11_68.npy', microstate_maps[:4, :7].T)