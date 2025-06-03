import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
data = np.load('data/91_original.npy')

# Prepare the result matrix
n_channels = data.shape[0]
gc_matrix = np.zeros((n_channels, n_channels))

# Perform Granger causality tests for each pair of channels
max_lag = 1  # You can adjust this based on your needs
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            test_result = grangercausalitytests(data[[i, j], :].T, maxlag=max_lag, verbose=False)
            # Get the p-value for the optimal lag
            p_values = [test_result[lag+1][0]['ssr_ftest'][1] for lag in range(max_lag)]
            gc_matrix[i, j] = min(p_values)  # Store the minimum p-value

# Save the results in (7,7) format
np.save('result/10_91.npy', gc_matrix)