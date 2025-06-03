import numpy as np
from sklearn.decomposition import FastICA

# Load the original data
data = np.load('data/41_original.npy')

# Check data shape and transpose if necessary
if data.shape[0] < data.shape[1]:
    data = data.T

# Perform ICA to extract independent components
ica = FastICA(n_components=7, random_state=0)
components = ica.fit_transform(data)

# Save the components (7 channels x time points)
np.save('result/7_41.npy', components.T)