import os
import numpy as np
from utils import extract_mfcc, create_directories
from tqdm import tqdm

DATASET_PATH = "../data/data/"

def extract_features():
    """Extract MFCC features from all audio files"""
    
    create_directories()
    
    features_ml = []  # For traditional ML (mean MFCCs)
    features_dl = []  # For deep learning (full MFCCs)
    labels = []
    file_paths = []
    
    print("Extracting MFCC features from AudioMNIST dataset...")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    speakers = sorted([d for d in os.listdir(DATASET_PATH) 
                      if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    print(f"Found {len(speakers)} speakers: {speakers}")
    
    # Process each speaker
    for speaker in tqdm(speakers, desc="Processing speakers"):
        speaker_folder = os.path.join(DATASET_PATH, speaker)
        
        audio_files = [f for f in os.listdir(speaker_folder) if f.endswith(".wav")]
        
        for file in tqdm(audio_files, desc=f"Speaker {speaker}", leave=False):
            file_path = os.path.join(speaker_folder, file)
            
            mfcc_mean, mfcc_full = extract_mfcc(file_path)
            
            if mfcc_mean is not None:
                features_ml.append(mfcc_mean)
                features_dl.append(mfcc_full)
                labels.append(int(speaker))
                file_paths.append(file_path)
    
    # Convert to numpy arrays
    features_ml = np.array(features_ml)
    features_dl = np.array(features_dl)
    labels = np.array(labels)
    
    print(f"\nExtraction complete!")
    print(f"Total samples: {len(labels)}")
    print(f"ML Features shape: {features_ml.shape}")
    print(f"DL Features shape: {features_dl.shape}")
    print(f"Unique speakers: {np.unique(labels)}")
    
    # Save features
    np.save("../features/mfcc_features_ml.npy", features_ml)
    np.save("../features/mfcc_features_dl.npy", features_dl)
    np.save("../features/labels.npy", labels)
    
    # Save metadata
    metadata = {
        'num_samples': len(labels),
        'num_speakers': len(np.unique(labels)),
        'speakers': list(np.unique(labels)),
        'ml_shape': features_ml.shape,
        'dl_shape': features_dl.shape
    }
    np.save("../features/metadata.npy", metadata)
    
    print("Features saved to ../features/ folder")
    return features_ml, features_dl, labels

if __name__ == "__main__":
    extract_features()