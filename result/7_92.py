import numpy as np
from sklearn.decomposition import FastICA

# Load the original data
data = np.load('data/92_original.npy')

# Check data shape and transpose if needed
if data.shape[0] < data.shape[1]:
    data = data.T

# Perform ICA to get independent components
ica = FastICA(n_components=7, random_state=0)
components = ica.fit_transform(data)

# Save the components
np.save('result/7_92.npy', components)