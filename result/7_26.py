
import numpy as np
from sklearn.decomposition import FastICA

# Load the original data
data = np.load('data/26_original.npy')

# Assuming the data shape is (channels, time_points), we transpose it for ICA
# ICA expects shape (n_samples, n_features)
data = data.T

# Perform ICA to extract independent components
ica = FastICA(n_components=7, random_state=0)
components = ica.fit_transform(data)  # This gives us the independent components

# Transpose back to get (components, time_points) format
components = components.T

# Save the result
np.save('result/7_26.npy', components)
