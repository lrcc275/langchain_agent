import numpy as np
from sklearn.decomposition import FastICA

def process_eeg_data(input_path, output_path, n_components=7):
    """
    Process EEG data by extracting independent components.
    
    Args:
        input_path: Path to input .npy file
        output_path: Path to save output .npy file
        n_components: Number of independent components to extract
    """
    # Load the original data
    data = np.load(input_path)
    
    # Check data shape and transpose if necessary (assuming channels are in rows)
    if data.shape[0] < data.shape[1]:
        data = data.T
    
    # Perform ICA
    ica = FastICA(n_components=n_components, random_state=0)
    components = ica.fit_transform(data)
    
    # Save the results
    np.save(output_path, components)
    
    print(f"Saved components with shape {components.shape} to {output_path}")

if __name__ == "__main__":
    process_eeg_data('data/59_original.npy', 'result/7_59.npy')
