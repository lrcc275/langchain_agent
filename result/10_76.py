
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
data = np.load('data/76_original.npy')

# Assuming the data shape is (n_channels, n_timepoints)
n_channels = data.shape[0]
gc_matrix = np.zeros((n_channels, n_channels))

# Perform Granger causality tests for all channel pairs
max_lag = 1  # You can adjust this based on your needs
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            test_result = grangercausalitytests(np.column_stack((data[i], data[j])), maxlag=max_lag, verbose=False)
            # Get the p-value for the optimal lag
            p_value = min([test_result[lag+1][0]['ssr_ftest'][1] for lag in range(max_lag)])
            gc_matrix[i, j] = p_value
        else:
            gc_matrix[i, j] = 0  # No causality for same channel

# Print the results
print("Granger Causality Matrix (p-values):")
print(gc_matrix)

# Ensure the matrix is (7,7)
if gc_matrix.shape != (7,7):
    raise ValueError("The resulting matrix is not (7,7)")

# Save the results
np.save('result/10_76.npy', gc_matrix)
