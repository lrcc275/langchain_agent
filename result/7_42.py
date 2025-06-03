import numpy as np
from sklearn.decomposition import FastICA

# Load the original data
data = np.load('data/42_original.npy')

# Extract independent components
n_components = min(data.shape[0], 7)
ica = FastICA(n_components=n_components, random_state=42)
components = ica.fit_transform(data.T).T

# Save the components
np.save('result/7_42.npy', components)