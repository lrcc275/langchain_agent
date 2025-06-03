import numpy as np
import mne
from sklearn.cluster import KMeans

# Load data
data = np.load('data/19_original.npy')

# Create info object
ch_names = [f'EEG{i+1}' for i in range(data.shape[0])]
info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')

# Create Raw object
raw = mne.io.RawArray(data, info)

# Preprocessing
raw.filter(1, 30, fir_design='firwin')

# Get data for clustering
data_for_clustering = raw.get_data().T

# Microstate analysis
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data_for_clustering)

# Get and reshape maps
maps = kmeans.cluster_centers_
if maps.shape[0] > 7:
    maps = maps[:7]
reshaped_maps = np.reshape(maps, (7,4))

# Save results
np.save('result/11_19.npy', reshaped_maps)