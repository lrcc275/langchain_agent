import numpy as np
from sklearn.cluster import KMeans

eeg_data = np.load("data/56_original.npy")
n_states = 4
kmeans = KMeans(n_clusters=n_states, random_state=42)
kmeans.fit(eeg_data.T)
microstate_maps = kmeans.cluster_centers_
if microstate_maps.shape[1] == 28:
    reshaped_maps = microstate_maps.reshape((n_states, 7, 4))
else:
    reshaped_maps = microstate_maps
np.save("result/11_56.npy", reshaped_maps)