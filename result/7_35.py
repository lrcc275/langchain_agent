
import numpy as np
from sklearn.decomposition import FastICA

# Load the data
data = np.load('data/35_original.npy')

# Perform ICA
ica = FastICA(n_components=data.shape[0], random_state=0)
components = ica.fit_transform(data.T).T

# Save the components
np.save('result/7_35.npy', components)
