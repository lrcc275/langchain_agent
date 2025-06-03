import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
data = np.load('data/55_original.npy')

# Assuming data shape is (n_channels, n_timepoints)
n_channels = data.shape[0]
gc_matrix = np.zeros((n_channels, n_channels))

# Perform pairwise Granger causality tests
max_lag = 1  # You can adjust this based on your needs
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            test_result = grangercausalitytests(data[[i, j], :].T, maxlag=max_lag, verbose=False)
            # Get the p-value for the optimal lag
            p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
            gc_matrix[i, j] = min(p_values)  # Store the most significant p-value

# Save the results
np.save('result/10_55.npy', gc_matrix)