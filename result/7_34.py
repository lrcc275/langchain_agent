import numpy as np
from sklearn.decomposition import FastICA

# Load the original data
data = np.load('data/34_original.npy')

# Perform ICA to extract independent components
ica = FastICA(n_components=7, random_state=0)
components = ica.fit_transform(data.T).T  # Transpose to get (components, time)

# Save the result
np.save('result/7_34.npy', components)