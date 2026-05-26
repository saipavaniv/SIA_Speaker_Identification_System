import librosa
import numpy as np
import os

def extract_mfcc(file_path, n_mfcc=13, max_pad_len=100):
    """
    Extract MFCC features from audio file
    
    Args:
        file_path: Path to audio file
        n_mfcc: Number of MFCC coefficients
        max_pad_len: Maximum length for padding
    
    Returns:
        Numpy array of MFCC features
    """
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length for consistency
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        # Flatten for traditional ML or keep 2D for DL
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean, mfcc
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['features', 'models', 'plots']
    for d in dirs:
        os.makedirs(f"../{d}", exist_ok=True)
    print("Directory structure verified.")