
import numpy as np
from sklearn.decomposition import FastICA

# Load the original data
data = np.load('data/29_original.npy')

# Perform ICA to extract independent components
ica = FastICA(n_components=data.shape[0], random_state=0)
components = ica.fit_transform(data.T).T

# Save the components in (7, x) format
np.save('result/7_29.npy', components[:7])
