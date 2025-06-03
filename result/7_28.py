import numpy as np
from sklearn.decomposition import FastICA

# Load the original data
data = np.load('data/28_original.npy')

# Perform ICA to extract independent components
ica = FastICA(n_components=data.shape[0])
components = ica.fit_transform(data.T).T

# Save the components
np.save('result/7_28.npy', components)