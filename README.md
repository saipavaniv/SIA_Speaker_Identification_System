# Speaker Identification System using MFCC Features

A comprehensive speaker identification system implementing both **traditional Machine Learning** and **Deep Learning** approaches using Mel Frequency Cepstral Coefficients (MFCCs) as features.

## 📋 Project Overview

This project implements a speaker identification system that can identify speakers from audio recordings using MFCC features. The system includes:

- **Traditional ML Models**: SVM, Random Forest, KNN
- **Deep Learning Model**: CNN (Convolutional Neural Network)
- **Ensemble Prediction**: Combines all models for robust predictions

## 🏗️ Project Structure

```
speaker-identification/
│
├── data/
│   └── AudioMNIST/
│       └── data/
│           ├── 01/  (speaker folders)
│           ├── 02/
│           └── ...
│
├── src/
│   ├── utils.py              # Feature extraction utilities
│   ├── extract_features.py   # Extract MFCC features from dataset
│   ├── train_model.py        # Train ML and DL models
│   ├── evaluate_model.py     # Evaluate all models
│   └── predict.py            # Predict speaker from audio file
│
├── features/                 # Extracted MFCC features (generated)
├── models/                   # Trained models (generated)
├── plots/                    # Visualization plots (generated)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd speaker-identification

# Install dependencies
pip install -r requirements.txt
```

### 2. Download AudioMNIST Dataset

Download the AudioMNIST dataset and place it in:
```
data/AudioMNIST/data/
```

The dataset should have folders named by speaker ID (e.g., 01, 02, 03, ..., 60).

### 3. Extract Features

```bash
cd src
python extract_features.py
```

This will:
- Extract MFCC features from all audio files
- Create two feature sets: one for ML models (mean MFCCs) and one for DL models (full MFCCs)
- Save features to `../features/` directory

### 4. Train Models

```bash
python train_model.py
```

This will train:
- SVM (Support Vector Machine)
- Random Forest
- KNN (K-Nearest Neighbors)
- CNN (Convolutional Neural Network)

All models are saved to `../models/` directory.

### 5. Evaluate Models

```bash
python evaluate_model.py
```

This will:
- Evaluate all models on test data
- Generate confusion matrices for each model
- Create comparison plots
- Print detailed metrics (accuracy, precision, recall, F1-score)
- Save all visualizations to `../plots/` directory

### 6. Predict Speaker from Audio

```bash
# Using ensemble (all models combined) - RECOMMENDED
python predict.py path/to/audio.wav

# Using specific model
python predict.py path/to/audio.wav svm
python predict.py path/to/audio.wav random_forest
python predict.py path/to/audio.wav knn
python predict.py path/to/audio.wav cnn
```

Example output:
```
Analyzing audio file: test_audio.wav
Using model: ENSEMBLE

==================================================
🎤 PREDICTED SPEAKER: 5
📊 CONFIDENCE: 94.32%
==================================================

This is Speaker 5 speaking.
```

## 📊 Models Overview

### Traditional Machine Learning Models

1. **SVM (Support Vector Machine)**
   - Kernel: RBF (Radial Basis Function)
   - Features: Mean MFCC coefficients
   - Fast prediction, good for small-medium datasets

2. **Random Forest**
   - Ensemble of 100 decision trees
   - Robust to overfitting
   - Handles non-linear patterns well

3. **KNN (K-Nearest Neighbors)**
   - K = 5 neighbors
   - Simple but effective
   - Distance-based classification

### Deep Learning Model

**CNN (Convolutional Neural Network)**
- Architecture:
  - 3 Convolutional blocks with BatchNorm and Dropout
  - Global Average Pooling
  - Dense layers with regularization
- Input: Full MFCC features (13 x 100 timesteps)
- Automatically learns hierarchical features
- Best performance for complex patterns

## 📈 Feature Extraction

**MFCC (Mel Frequency Cepstral Coefficients)**:
- 13 MFCC coefficients extracted per audio frame
- Sampling rate: 16 kHz
- Represents spectral characteristics of speech
- Captures speaker-specific vocal tract properties

**Two feature representations**:
1. **Mean MFCCs** (for ML): Average across time (13 features)
2. **Full MFCCs** (for DL): All timesteps (13 x 100 matrix)

## 📊 Evaluation Metrics

The system evaluates models using:
- **Accuracy**: Overall correct predictions
- **Precision**: Correct positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

## 🎯 Expected Performance

On AudioMNIST dataset (approximate):
- SVM: 85-92% accuracy
- Random Forest: 83-90% accuracy
- KNN: 80-88% accuracy
- CNN: 90-96% accuracy
- Ensemble: 92-97% accuracy

*Note: Actual performance depends on dataset size and quality*

## 🔧 Customization

### Change MFCC Parameters

In `utils.py`:
```python
def extract_mfcc(file_path, n_mfcc=13, max_pad_len=100):
    # Adjust n_mfcc for more/fewer coefficients
    # Adjust max_pad_len for different audio lengths
```

### Modify CNN Architecture

In `train_model.py`:
```python
def build_cnn_model(input_shape, num_classes):
    # Modify layers, filters, dropout rates
```

## 🐛 Troubleshooting

**Issue: "Dataset not found"**
- Ensure AudioMNIST data is in `data/AudioMNIST/data/`
- Check folder structure matches expected format

**Issue: "Model file not found"**
- Run `extract_features.py` before training
- Run `train_model.py` before evaluation/prediction

**Issue: Low accuracy**
- Increase training data
- Adjust model hyperparameters
- Try different MFCC parameters

## 📚 References

1. [AudioMNIST Dataset](https://github.com/soerenab/AudioMNIST)
2. [Librosa Documentation](https://librosa.org/doc/latest/index.html)
3. [MFCC Feature Extraction](https://librosa.org/doc/main/generated/librosa.feature.mfcc.html)
4. [Speaker Identification Research](https://ieeexplore.ieee.org/document/speaker-recognition)

## 📝 Assignment Requirements Checklist

✅ Extract MFCCs from speech samples  
✅ Train machine learning models (SVM, RF, KNN)  
✅ Train deep learning model (CNN)  
✅ Evaluate with accuracy, precision, recall, F1-score  
✅ Generate confusion matrices  
✅ Use AudioMNIST dataset  
✅ Implement speaker prediction system  
✅ Comprehensive documentation  

## 👥 Author

Your Name - Sai Pavani V