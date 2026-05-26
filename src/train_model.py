import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

def train_ml_models():
    """Train traditional ML models (SVM, RF, KNN)"""
    print("\n" + "="*50)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*50)
    
    # Load features
    print("\nLoading features...")
    features = np.load("../features/mfcc_features_ml.npy")
    labels = np.load("../features/labels.npy")
    
    print(f"Features shape: {features.shape}")
    print(f"Number of samples: {len(labels)}")
    print(f"Number of speakers: {len(np.unique(labels))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, "../models/scaler.pkl")
    
    results = {}
    
    # Train SVM
    print("\n" + "-"*50)
    print("1. Training SVM (Support Vector Machine)...")
    print("-"*50)
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"✓ SVM Accuracy: {svm_acc:.4f} ({svm_acc*100:.2f}%)")
    joblib.dump(svm_model, "../models/svm_model.pkl")
    results['svm'] = float(svm_acc)
    
    # Train Random Forest
    print("\n" + "-"*50)
    print("2. Training Random Forest...")
    print("-"*50)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"✓ Random Forest Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    joblib.dump(rf_model, "../models/rf_model.pkl")
    results['random_forest'] = float(rf_acc)
    
    # Train KNN
    print("\n" + "-"*50)
    print("3. Training KNN (K-Nearest Neighbors)...")
    print("-"*50)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)
    print(f"✓ KNN Accuracy: {knn_acc:.4f} ({knn_acc*100:.2f}%)")
    joblib.dump(knn_model, "../models/knn_model.pkl")
    results['knn'] = float(knn_acc)
    
    # Save test data for evaluation
    np.save("../features/X_test_ml.npy", X_test)
    np.save("../features/y_test.npy", y_test)
    
    return results

def main():
    """Main training pipeline"""
    print("\n🎤 Starting Model Training Pipeline...")
    
    # Train ML models
    results = train_ml_models()
    
    # Save results
    with open("../models/training_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for model_name, acc in results.items():
        print(f"{model_name.upper():<20} {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n✅ All models saved to ../models/ folder")
    print("✅ Training complete!")
    
    print("\n💡 Next steps:")
    print("   1. Run: python evaluate_model.py")
    print("   2. Run: python predict.py <audio_file.wav>")

if __name__ == "__main__":
    main()