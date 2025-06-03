import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
data = np.load('data/58_original.npy')
n_channels = data.shape[0]

# Initialize results matrix
gc_matrix = np.zeros((n_channels, n_channels))

# Perform pairwise Granger causality tests
max_lag = 1  # You can adjust this based on your needs
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            test_result = grangercausalitytests(data[[i, j], :].T, maxlag=max_lag, verbose=False)
            # Get the p-value for the optimal lag
            p_value = test_result[max_lag][0]['ssr_ftest'][1]
            gc_matrix[i, j] = p_value

# Save the results
np.save('result/10_58.npy', gc_matrix)