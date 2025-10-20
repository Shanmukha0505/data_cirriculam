
# Create evaluation.py
evaluation_code = '''"""
Model Evaluation Module
Generates confusion matrices, metrics, and performance visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate classification metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        Class labels
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()


def print_classification_report(y_true, y_pred, labels=None):
    """
    Print detailed classification report
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        Class labels
    """
    print("\\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))


def compare_models(results_dict):
    """
    Compare multiple models and display results
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metrics as values
        
    Returns:
    --------
    pandas.DataFrame
        Comparison table
    """
    df_results = pd.DataFrame(results_dict).T
    df_results = df_results.sort_values('f1_score', ascending=False)
    
    print("\\nModel Comparison:")
    print("=" * 80)
    print(df_results.to_string())
    
    return df_results


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of features
    top_n : int
        Number of top features to display
    save_path : str, optional
        Path to save figure
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    
    plt.close()


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models and return metrics
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
        
    Returns:
    --------
    dict
        Dictionary of metrics for each model
    """
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred, model_name=name)
        results[name] = metrics
        
        print(f"\\n{name.upper()} Results:")
        print("-" * 60)
        for metric, value in metrics.items():
            if metric != 'model':
                print(f"{metric.capitalize()}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
'''

with open('data_curriculum/src/evaluation.py', 'w') as f:
    f.write(evaluation_code)

print("Created: src/evaluation.py")
