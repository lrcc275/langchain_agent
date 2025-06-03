import numpy as np
from sklearn.decomposition import FastICA

# Load the original data
data = np.load('data/72_original.npy')

# Assuming data shape is (channels, time_points)
n_components = data.shape[0]  # Number of channels (7)
ica = FastICA(n_components=n_components)
components = ica.fit_transform(data.T).T  # Transpose to get correct shape

# Save the components
np.save('result/7_72.npy', components)