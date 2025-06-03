import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
data = np.load('data/70_original.npy')

# Prepare the result matrix
n_channels = data.shape[0]
result_matrix = np.zeros((n_channels, n_channels))

# Perform Granger causality tests for each pair of channels
max_lag = 1  # You can adjust this based on your needs
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            test_result = grangercausalitytests(data[[i, j], :].T, maxlag=max_lag, verbose=False)
            # Get the p-value for the optimal lag (using lag 1 here)
            p_value = test_result[max_lag][0]['ssr_ftest'][1]
            result_matrix[i, j] = p_value

# Save the result
np.save('result/10_70.npy', result_matrix)
