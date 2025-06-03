import numpy as np
from sklearn.decomposition import FastICA

data = np.load('data/91_original.npy')
if data.shape[0] < data.shape[1]:
    data = data.T

ica = FastICA(n_components=7, random_state=0)
components = ica.fit_transform(data).T
np.save('result/7_91.npy', components)