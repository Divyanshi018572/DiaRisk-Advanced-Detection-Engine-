import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to sys.path
# This assumes the script is run from the project root or pipeline directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def run_eda():
    """Runs data exploration and saves plots to outputs folder."""
    # 1. Load Data
    csv_file = path_utils.get_data_path("raw", "diabetes_binary_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(csv_file)
    print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Null Values: {df.isnull().sum().sum()}")
    
    # Ensure outputs directory exists
    os.makedirs(path_utils.get_outputs_path(), exist_ok=True)
    
    # 2. Target Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Diabetes_binary', data=df, palette='viridis')
    plt.title('Diabetes Distribution (0=No, 1=Yes/Pre)')
    plt.savefig(path_utils.get_outputs_path("diabetes_distribution.png"))
    plt.close()
    
    # 3. Diabetes Rate by Age Category
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Age', y='Diabetes_binary', data=df, marker='o', color='blue')
    plt.title('Diabetes Rate by Age category (1=18-24, 13=80+)')
    plt.ylabel('Risk Probability')
    plt.grid(True)
    plt.savefig(path_utils.get_outputs_path("diabetes_rate_by_age.png"))
    plt.close()
    
    # 4. Diabetes Rate by BMI Bins
    # (Underweight <18.5, Normal 18.5–25, Overweight 25–30, Obese 30+)
    df['BMI_Bins'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], 
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='BMI_Bins', y='Diabetes_binary', data=df, palette='coolwarm')
    plt.title('Diabetes Rate by BMI Category')
    plt.ylabel('Risk Probability')
    plt.savefig(path_utils.get_outputs_path("diabetes_rate_by_bmi.png"))
    plt.close()
    
    # 5. Diabetes Rate by GenHlth (General Health Rating 1-5)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='GenHlth', y='Diabetes_binary', data=df, palette='OrRd')
    plt.title('Diabetes Rate by General Health Rating (1=Excellent, 5=Poor)')
    plt.ylabel('Risk Probability')
    plt.savefig(path_utils.get_outputs_path("diabetes_rate_by_genhlth.png"))
    plt.close()
    
    # 6. Correlation Heatmap (Top 10 features correlate with Diabetes)
    plt.figure(figsize=(12, 10))
    # Filter only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    # Pull Top Correlations with Target
    top_corr = corr_matrix['Diabetes_binary'].sort_values(ascending=False).head(10).index
    sns.heatmap(df[top_corr].corr(), annot=True, cmap='RdBu_r', center=0)
    plt.title('Top 10 Feature Correlation Heatmap')
    plt.savefig(path_utils.get_outputs_path("correlation_heatmap.png"))
    plt.close()
    
    # 7. Comparison: Mean values for Diabetic vs Non-Diabetic
    comp_features = ['HighBP', 'HighChol', 'BMI', 'Age', 'Smoker', 'HeartDiseaseorAttack', 'PhysActivity']
    comp_df = df.groupby('Diabetes_binary')[comp_features].mean().reset_index()
    # Melt for plotting
    comp_melted = comp_df.melt(id_vars='Diabetes_binary', var_name='Feature', value_name='Mean Value')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Feature', y='Mean Value', hue='Diabetes_binary', data=comp_melted, palette='muted')
    plt.title('Mean Feature Values: Diabetic (1) vs Non-Diabetic (0)')
    plt.xticks(rotation=45)
    plt.savefig(path_utils.get_outputs_path("feature_means_comparison.png"))
    plt.close()
    
    print("EDA Visualizations saved to 'outputs/'")

if __name__ == "__main__":
    run_eda()
