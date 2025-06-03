import numpy as np
from pathlib import Path
import os

# Define paths
input_path = "data/13_original.npy"
output_npy_path = "result/1_13.npy"

# Create result directory if it doesn't exist
Path("result").mkdir(exist_ok=True)

# Check if input file exists
if not os.path.exists(input_path):
    print(f"Error: Input file {input_path} not found")
    # Create empty results file
    np.save(output_npy_path, np.zeros((7, 3)))
else:
    # Load data
    data = np.load(input_path)
    
    # Calculate metrics for each channel (assuming channels are rows)
    peak_to_peak = np.ptp(data, axis=1)
    means = np.mean(data, axis=1)
    variances = np.var(data, axis=1)
    
    # Combine results into (7, 3) array
    results = np.column_stack((peak_to_peak, means, variances))
    
    # Print results
    print("Channel\tPeak-to-Peak\tMean\t\tVariance")
    for i, (pp, mean, var) in enumerate(zip(peak_to_peak, means, variances)):
        print(f"{i+1}\t{pp:.3f}\t\t{mean:.3f}\t\t{var:.3f}")
    
    # Save results
    np.save(output_npy_path, results)