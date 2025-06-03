import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
data = np.load('data/25_original.npy')

# Assuming the data shape is (channels, time_points)
n_channels = data.shape[0]
gc_matrix = np.zeros((n_channels, n_channels))

# Perform Granger causality tests for each pair of channels
max_lag = 1  # You can adjust this based on your needs
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            test_result = grangercausalitytests(data[[i, j], :].T, maxlag=max_lag, verbose=False)
            # Get the F-test p-value for the optimal lag
            p_value = test_result[max_lag][0]['ssr_ftest'][1]
            gc_matrix[i, j] = p_value

# Save the results in (7,7) format
np.save('result/10_25.npy', gc_matrix[:7, :7])