import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve, confusion_matrix

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def run_evaluation():
    """Evaluates all models including the Elite Stacked Ensemble."""
    # 1. Load Data
    X_test = pd.read_csv(path_utils.get_data_path("processed", "X_test.csv"))
    y_test = pd.read_csv(path_utils.get_data_path("processed", "y_test.csv")).values.ravel()
    
    # 2. Identify Models and Load Them
    # Add 'Elite Ensemble' and 'XGBoost (Accuracy Opt)' to comparison
    model_paths = {
        'Logistic Regression': "logistic_regression.pkl",
        'Naive Bayes': "naive_bayes.pkl",
        'Random Forest': "random_forest.pkl",
        'Elite Ensemble (Max Accuracy)': "elite_ensemble.pkl",
        'XGBoost (Recall Opt)': "xgboost_model.pkl"
    }
    
    models = {}
    for name, filename in model_paths.items():
        path = path_utils.get_models_path(filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"Warning: {filename} not found.")
            
    # 3. Store Results
    results = []
    plt.figure(figsize=(12, 10))
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        # Get Probabilities for ROC curve
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # Stacking classifier has predict_proba
            y_prob = model.decision_function(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'AUC-ROC': auc,
            'Recall': rec,
            'F1-Score': f1
        })
        
        # ROC Plotting
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    
    # Finalize ROC Plot
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Comparing Baseline vs Elite Ensemble')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(path_utils.get_outputs_path("roc_curves_elite_comparison.png"))
    plt.close()
    
    # 4. Save Performance Table
    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    print("\nElite Model Performance Comparison:")
    print(results_df)
    results_df.to_csv(path_utils.get_outputs_path("performance_metrics_elite.csv"), index=False)
    
    # 5. Elite Confusion Matrix
    if 'Elite Ensemble (Max Accuracy)' in models:
        best_model = models['Elite Ensemble (Max Accuracy)']
        y_pred_elite = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_elite)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
        plt.title('Confusion Matrix: Elite Stacked Ensemble')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(path_utils.get_outputs_path("confusion_matrix_elite.png"))
        plt.close()
    
    print("Elite Evaluation Complete.")

if __name__ == "__main__":
    run_evaluation()
