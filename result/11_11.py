import numpy as np
from sklearn.cluster import KMeans

# Load data
eeg_data = np.load('data/11_original.npy')

# Microstate analysis
n_microstates = 7
kmeans = KMeans(n_clusters=n_microstates, random_state=42)
microstates = kmeans.fit_predict(eeg_data.T)

# Prepare and save results
result = np.zeros((7,4))
for i in range(7):
    for j in range(4):
        result[i,j] = np.mean(eeg_data[i*4+j] if i*4+j < eeg_data.shape[0] else 0)
np.save('result/11_11.npy', result)