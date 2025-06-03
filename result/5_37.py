
import numpy as np
from antropy import sample_entropy, app_entropy, spectral_entropy
from scipy.signal import welch
import os

def process_eeg_data(input_path, output_path, fs=250, segment_length=10):
    # Load data
    data = np.load(input_path)
    segment_samples = segment_length * fs
    nperseg = 256  # Fixed number of points for PSD calculation
    
    # Calculate number of segments
    n_segments = data.shape[1] // segment_samples
    
    # Initialize result array (7 channels, 3 entropy measures, n_segments)
    results = np.zeros((7, 3, n_segments))
    
    # Process each segment
    for seg in range(n_segments):
        start = seg * segment_samples
        end = start + segment_samples
        segment = data[:, start:end]
        
        for ch in range(7):  # 7 channels
            # Sample entropy
            results[ch, 0, seg] = sample_entropy(segment[ch], order=2)
            
            # Approximate entropy
            results[ch, 1, seg] = app_entropy(segment[ch], order=2)
            
            # Spectral entropy with fixed parameters
            freqs, psd = welch(segment[ch], fs=fs, nperseg=nperseg)
            results[ch, 2, seg] = spectral_entropy(psd, sf=fs, normalize=True)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, results)
    return results

if __name__ == "__main__":
    results = process_eeg_data('data/37_original.npy', 'result/5_37.npy')
    print("Entropy measures for each channel and segment:")
    print(results)
