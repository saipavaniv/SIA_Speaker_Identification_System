import joblib
import numpy as np
from utils import extract_mfcc
import sys
import os

def predict_with_ml(audio_path, model_name='svm'):
    """Predict speaker using ML model"""
    # Load model and scaler
    model = joblib.load(f"../models/{model_name}_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    
    # Extract features
    mfcc_mean, _ = extract_mfcc(audio_path)
    
    if mfcc_mean is None:
        return None, None
    
    # Normalize
    mfcc_normalized = scaler.transform([mfcc_mean])
    
    # Predict
    prediction = model.predict(mfcc_normalized)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(mfcc_normalized)[0]
        confidence = np.max(probabilities)
    else:
        confidence = None
    
    return prediction, confidence

def predict_ensemble(audio_path):
    """Predict using ensemble of all ML models"""
    predictions = []
    confidences = []
    
    # All ML models
    for model_name in ['svm', 'random_forest', 'knn']:
        try:
            pred, conf = predict_with_ml(audio_path, model_name)
            if pred is not None:
                predictions.append(pred)
                if conf is not None:
                    confidences.append(conf)
        except Exception as e:
            print(f"Warning: Could not use {model_name} model: {e}")
    
    # Majority vote
    if predictions:
        prediction = int(np.bincount(predictions).argmax())
        avg_confidence = np.mean(confidences) if confidences else None
        return prediction, avg_confidence
    
    return None, None

def main():
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("Usage: python predict.py <audio_file_path> [model_type]")
        print("="*60)
        print("\nModel type options:")
        print("  - svm           : Support Vector Machine")
        print("  - random_forest : Random Forest")
        print("  - knn           : K-Nearest Neighbors")
        print("  - ensemble      : All models combined (default)")
        print("\nExample:")
        print("  python predict.py ../data/AudioMNIST/data/05/05_01_0.wav")
        print("  python predict.py test.wav svm")
        print()
        sys.exit(1)
    
    audio_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'ensemble'
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"\n❌ Error: File '{audio_path}' not found!")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"🎤 SPEAKER IDENTIFICATION SYSTEM")
    print(f"{'='*60}")
    print(f"\n📁 Audio file: {audio_path}")
    print(f"🤖 Model: {model_type.upper()}")
    print(f"\n{'='*60}")
    print("Processing...")
    print(f"{'='*60}")
    
    # Predict based on model type
    try:
        if model_type == 'ensemble':
            prediction, confidence = predict_ensemble(audio_path)
        elif model_type in ['svm', 'random_forest', 'knn']:
            prediction, confidence = predict_with_ml(audio_path, model_type)
        else:
            print(f"\n❌ Error: Unknown model type '{model_type}'")
            print("Available models: svm, random_forest, knn, ensemble")
            sys.exit(1)
        
        # Display results
        if prediction is not None:
            print(f"\n{'='*60}")
            print(f"🎉 PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"\n🎤 PREDICTED SPEAKER: {prediction}")
            if confidence is not None:
                print(f"📊 CONFIDENCE: {confidence:.2%}")
            print(f"\n{'='*60}")
            print(f"✅ This is Speaker {prediction} speaking.")
            print(f"{'='*60}\n")
        else:
            print("\n❌ Error: Could not process audio file.")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: Model file not found!")
        print("Please run 'python train_model.py' first to train the models.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()