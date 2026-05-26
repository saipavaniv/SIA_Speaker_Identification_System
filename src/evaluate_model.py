import numpy as np
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_confusion_matrix(cm, title, filename, labels):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Speaker', fontsize=12)
    plt.xlabel('Predicted Speaker', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"../plots/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: ../plots/{filename}")

def evaluate_ml_model(model_name, model_path):
    """Evaluate a traditional ML model"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load model and data
    model = joblib.load(model_path)
    X_test = np.load("../features/X_test_ml.npy")
    y_test = np.load("../features/y_test.npy")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    print(f"\n{model_name.upper()} Results:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(np.unique(y_test))
    plot_confusion_matrix(
        cm, 
        f'{model_name.upper()} - Confusion Matrix',
        f'confusion_matrix_{model_name}.png',
        labels
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }

def plot_model_comparison(results):
    """Plot comparison of all models"""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        bars = axes[idx].bar(models, values, color=colors)
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, pad=10)
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10
            )
    
    plt.tight_layout()
    plt.savefig('../plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: ../plots/model_comparison.png")

def main():
    """Main evaluation pipeline"""
    print("\n" + "🎤"*30)
    print("MODEL EVALUATION")
    print("🎤"*30 + "\n")
    
    results = {}
    
    # Evaluate ML models
    ml_models = {
        'svm': '../models/svm_model.pkl',
        'random_forest': '../models/rf_model.pkl',
        'knn': '../models/knn_model.pkl'
    }
    
    for name, path in ml_models.items():
        results[name] = evaluate_ml_model(name, path)
    
    # Save results
    with open("../models/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Plot comparison
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOTS")
    print("="*60)
    plot_model_comparison(results)
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\n{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    for model, metrics in results.items():
        print(f"{model.upper():<15} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0].upper()} with accuracy {best_model[1]['accuracy']:.4f}")
    
    print("\n✅ Evaluation complete! All plots saved to ../plots/")

if __name__ == "__main__":
    main()