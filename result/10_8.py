import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
data = np.load('data/8_original.npy')

# Prepare the result matrix (7x7)
n_channels = data.shape[0]
gc_matrix = np.zeros((n_channels, n_channels))

# Perform Granger causality tests for each pair of channels
max_lag = 2  # You can adjust this based on your needs
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            test_result = grangercausalitytests(data[[i, j], :].T, maxlag=max_lag, verbose=False)
            # Get the minimum p-value across all tested lags
            p_values = [test_result[lag+1][0]['ssr_ftest'][1] for lag in range(max_lag)]
            gc_matrix[i, j] = min(p_values)

# Save the results
np.save('result/10_8.npy', gc_matrix)
