import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def run_training():
    """Trains baseline models and a champion Elite Stacked Ensemble for maximum accuracy."""
    # 1. Load Processed Data
    X_train = pd.read_csv(path_utils.get_data_path("processed", "X_train.csv"))
    y_train = pd.read_csv(path_utils.get_data_path("processed", "y_train.csv")).values.ravel()
    
    print(f"HF-Optimized Training on {X_train.shape[0]} samples with {X_train.shape[1]} features.")
    
    # 2. Train Optimized Baselines
    print("Training Optimized Baselines (Pruned to <10MB)...")
    
    # Logistic Regression (Tiny)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42).fit(X_train, y_train)
    joblib.dump(lr, path_utils.get_models_path("logistic_regression.pkl"), compress=3)
    
    # Naive Bayes (Tiny)
    nb = GaussianNB(priors=[0.86, 0.14], var_smoothing=1e-8).fit(X_train, y_train)
    joblib.dump(nb, path_utils.get_models_path("naive_bayes.pkl"), compress=3)
    
    # 🛑 KNN Removed: Too large for standard HF push (>60MB)
    if os.path.exists(path_utils.get_models_path("knn.pkl")):
        os.remove(path_utils.get_models_path("knn.pkl"))
    
    # Pruning Random Forest to < 10MB (max_depth=10, n_estimators=75)
    rf = RandomForestClassifier(
        n_estimators=75, max_depth=10, class_weight='balanced', 
        random_state=42, n_jobs=-1
    ).fit(X_train, y_train)
    joblib.dump(rf, path_utils.get_models_path("random_forest.pkl"), compress=3)
    
    # 3. Define "Elite" Base Learners (Pruned for 10MB threshold)
    print("Initializing Elite Ensemble Base Learners (Pruned for Deploy)...")
    
    xgb_elite = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, 
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    
    lgb_elite = LGBMClassifier(
        n_estimators=100, num_leaves=31, learning_rate=0.1, 
        random_state=42, verbose=-1
    )
    
    cat_elite = CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1, 
        random_seed=42, verbose=0, allow_writing_files=False
    )
    
    rf_elite = RandomForestClassifier(
        n_estimators=80, max_depth=10, random_state=42, n_jobs=-1
    )

    # 4. Create Stacked Ensemble
    print("Training Elite Stacked Ensemble (XGB + LGBM + Cat + RF)...")
    base_learners = [
        ('xgb', xgb_elite),
        ('lgbm', lgb_elite),
        ('cat', cat_elite),
        ('rf', rf_elite)
    ]
    
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
        n_jobs=-1
    )
    
    stack.fit(X_train, y_train)
    # Save with Level 3 Compression (Ensuring < 10MB)
    joblib.dump(stack, path_utils.get_models_path("elite_ensemble.pkl"), compress=3)
    
    # 5. XGBoost (Recall Optimized) - Tiny
    xgb_recall = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, 
        scale_pos_weight=6.14, random_state=42, eval_metric='logloss'
    ).fit(X_train, y_train)
    joblib.dump(xgb_recall, path_utils.get_models_path("xgboost_model.pkl"), compress=3)
    
    print("10MB Threshold Training Complete. All models saved successfully.")

if __name__ == "__main__":
    run_training()
