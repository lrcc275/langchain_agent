
import numpy as np
from sklearn.decomposition import FastICA

# Load the original data
data = np.load('data/90_original.npy')

# Assuming data shape is (channels, time_points), perform ICA
ica = FastICA(n_components=7, random_state=0)  # Extract 7 independent components
components = ica.fit_transform(data.T).T  # Transpose to fit ICA and transpose back

# Save the components to result/7_90.npy
np.save('result/7_90.npy', components)
