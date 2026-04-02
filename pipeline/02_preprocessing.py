import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import joblib

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def run_preprocessing():
    """Performs feature engineering, scaling, and splitting."""
    # 1. Load Data
    csv_file = path_utils.get_data_path("raw", "diabetes_binary_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(csv_file)
    print(f"Initial Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Clinical Feature Engineering
    print("Engineering Clinical Features...")
    # BMI_OBESE = 1 if BMI >= 30 else 0
    df['BMI_OBESE'] = (df['BMI'] >= 30).astype(int)
    
    # HIGH_RISK_COMBO = 1 if HighBP == 1 AND HighChol == 1 else 0
    df['HIGH_RISK_COMBO'] = ((df['HighBP'] == 1) & (df['HighChol'] == 1)).astype(int)
    
    # POOR_HEALTH_SCORE = GenHlth + DiffWalk + PhysHlth_flag
    # where PhysHlth_flag = 1 if PhysHlth > 14
    df['PhysHlth_flag'] = (df['PhysHlth'] > 14).astype(int)
    df['POOR_HEALTH_SCORE'] = df['GenHlth'] + df['DiffWalk'] + df['PhysHlth_flag']
    
    # Drop intermediate flag
    df.drop(columns=['PhysHlth_flag'], inplace=True)
    
    print(f"Features Engineered. Total columns: {df.shape[1]}")
    
    # 3. Features and Target
    X = df.drop(columns=['Diabetes_binary'])
    y = df['Diabetes_binary']
    
    # 4. Train-Test Split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Splits Created: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # 5. Scaling
    print("Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler for later use in app.py
    os.makedirs(path_utils.get_models_path(), exist_ok=True)
    joblib.dump(scaler, path_utils.get_models_path("scaler.pkl"))
    
    # 6. Save Processed Data
    os.makedirs(path_utils.get_data_path("processed"), exist_ok=True)
    
    # Convert scaled back to DataFrame to preserve feature names for training script
    X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    X_train_final.to_csv(path_utils.get_data_path("processed", "X_train.csv"), index=False)
    X_test_final.to_csv(path_utils.get_data_path("processed", "X_test.csv"), index=False)
    y_train.to_csv(path_utils.get_data_path("processed", "y_train.csv"), index=False)
    y_test.to_csv(path_utils.get_data_path("processed", "y_test.csv"), index=False)
    
    print("Preprocessing Complete. Data and Scaler saved.")

if __name__ == "__main__":
    run_preprocessing()
