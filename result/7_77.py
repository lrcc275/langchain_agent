import numpy as np
from sklearn.decomposition import FastICA

# Load the original data
data = np.load('data/77_original.npy')

# Perform ICA to extract 7 independent components
ica = FastICA(n_components=7, random_state=0)
components = ica.fit_transform(data.T).T  # Transpose to get (7, x) shape

# Save the components
np.save('result/7_77.npy', components)