import numpy as np
from scipy import signal
import os

# Create result directory if it doesn't exist
os.makedirs('result', exist_ok=True)

try:
    # Load the data
    data = np.load('data/63_original.npy')
    
    # Parameters
    fs = 250  # Assuming sampling rate is 250Hz (common for EEG)
    nperseg = 4 * fs  # 4 second window
    noverlap = nperseg // 2  # 50% overlap

    # Compute PSD for each channel
    psd_results = []
    for channel in data:
        f, Pxx = signal.welch(channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
        psd_results.append(Pxx)

    psd_results = np.array(psd_results)

    # Print results
    print(psd_results)

    # Save results
    np.save('result/3_63.npy', psd_results)

except FileNotFoundError:
    print("Error: The file data/63_original.npy was not found.")
    # Create empty file to indicate error
    np.save('result/3_63.npy', np.array([]))