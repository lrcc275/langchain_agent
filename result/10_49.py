import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
data = np.load('data/49_original.npy')

# Perform Granger causality analysis for each pair of channels
n_channels = data.shape[0]
gc_matrix = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            # Prepare the data for Granger test (needs to be 2D array with 2 columns)
            test_data = np.column_stack((data[i], data[j]))
            # Perform Granger test with maxlag (you can adjust this)
            gc_result = grangercausalitytests(test_data, maxlag=1, verbose=False)
            # Get the p-value for lag 1
            p_value = gc_result[1][0]['ssr_ftest'][1]
            gc_matrix[i, j] = p_value
        else:
            gc_matrix[i, j] = 0  # No causality for same channel

# Save the results
np.save('result/10_49.npy', gc_matrix)