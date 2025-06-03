import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
data = np.load('data/1_original.npy')

# Initialize the result matrix (7x7)
n_channels = 7
gc_matrix = np.zeros((n_channels, n_channels))

# Perform pairwise Granger causality tests
max_lag = 1  # You can adjust this based on your needs
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            # Prepare the data for the test
            test_data = np.column_stack((data[:, i], data[:, j]))
            # Perform Granger test
            gc_result = grangercausalitytests(test_data, max_lag, verbose=False)
            # Store the p-value for lag 1
            p_value = gc_result[1][0]['ssr_ftest'][1]
            gc_matrix[i, j] = p_value

# Save the results
np.save('result/10_1.npy', gc_matrix)