import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Path management
import path_utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DiaRisk-Advanced Detection Engine",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM NAVY DARK THEME CSS ---
st.markdown("""
<style>
    /* Main Background & Text */
    .main {
        background-color: #0c121c;
        color: #ffffff;
    }
    
    /* Global Text Color */
    html, body, [class*="st-"] {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #16213e;
        border-right: 1px solid #1f4068;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2ec4b6 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input Fields Border & Background */
    div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"] {
        background-color: #1b263b !important;
        border: 1px solid #334e68 !important;
        border-radius: 8px !important;
    }
    
    /* Checkbox & Radio Labels */
    .stCheckbox label, .stRadio label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%);
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(67, 97, 238, 0.5);
        color: #ffffff;
    }
    
    /* Custom Risk Cards */
    .risk-card {
        padding: 25px;
        border-radius: 15px;
        background: rgba(22, 33, 62, 0.8);
        border: 1px solid #334e68;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .driver-item {
        font-size: 1.05em;
        padding: 10px;
        margin-bottom: 5px;
        background: rgba(27, 38, 59, 0.6);
        border-radius: 8px;
        border-left: 4px solid #4361ee;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0c121c;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #16213e;
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1f4068;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    models = {
        '⭐ Elite Ensemble (Max Accuracy)': joblib.load(path_utils.get_models_path("elite_ensemble.pkl")),
        'XGBoost (Recall Optimized)': joblib.load(path_utils.get_models_path("xgboost_model.pkl")),
        'Logistic Regression': joblib.load(path_utils.get_models_path("logistic_regression.pkl")),
        'Random Forest': joblib.load(path_utils.get_models_path("random_forest.pkl")),
        'Naive Bayes': joblib.load(path_utils.get_models_path("naive_bayes.pkl"))
    }
    scaler = joblib.load(path_utils.get_models_path("scaler.pkl"))
    return models, scaler

models, scaler = load_assets()

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/color/96/diabetes.png", width=100)
st.sidebar.title("App Intelligence")
selected_model_name = st.sidebar.selectbox("Predictive Engine", list(models.keys()))
threshold = st.sidebar.slider("Risk Cutoff Threshold", 0.3, 0.7, 0.5, 0.05)
st.sidebar.info("Adjust threshold to balance medical sensitivity vs precision.")

st.sidebar.divider()
st.sidebar.markdown("### 📊 Project Insights")
st.sidebar.write("""
This tool analyzes 22 health indicators from the CDC BRFSS dataset (253k rows) to quantify Diabetes risk.
""")

# --- TABS ---
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Project Overview",
    "🏥 Patient Risk Portal", 
    "📊 Population Explorer", 
    "📈 Model Calibration", 
    "⚖️ Disclaimer"
])

# --- TAB 0: PROJECT OVERVIEW ---
with tab0:
    st.title("🛡️ DiaRisk-Advanced Detection Engine")
    st.markdown("""
    **DiaRisk** is a professional-grade clinical detection engine that transforms 253k+ CDC records into actionable health insights. 
    By leveraging a high-recall XGBoost and an Elite Stacked Ensemble, it identifies Type 2 Diabetes risk patterns with **86.4% accuracy**. 
    Designed for clinical pre-screening, it empowers early intervention through expert analytics and real-time risk stratification.
    """)
    
    col_ov1, col_ov2 = st.columns([1, 1])
    with col_ov1:
        st.subheader("The Problem Statement")
        st.info("""
        **Mission:** Identify high-risk individuals for Type 2 Diabetes using lifestyle and demographic indicators.
        
        **Challenge:** How do we balance 'False Alarms' (Precision) vs. 'Missing Patients' (Recall)? 
        In clinical environments, **Missing a diabetic case is 10x more costly than a false alarm.**
        """)
        
    with col_ov2:
        st.subheader("Performance Strategy")
        st.write("""
        We implemented two distinct model philosophies:
        1. **Precision Elite (Stacked Ensemble):** Maximizes global Accuracy (86.4%).
        2. **Clinical Heavyweight (Recall Opt XGB):** Maximizes detection of patients (79% Recall).
        """)

    st.divider()
    
    st.subheader("🏆 The 'Best Overall' Model Analysis")
    st.markdown("""
    According to the clinical problem statement, the **XGBoost (Recall Optimized)** is the best overall performer.
    
    | Model Tier | Metric Focus | Clinical Value |
    |:---|:---|:---|
    | **XGBoost (Recall Opt)** | **79% Sensitivity** | **High Utility** - Highest safety net for patient screening. |
    | **Elite Ensemble** | **86.4% Accuracy** | **Technical Excellence** - Best for population-wide statistics. |
    
    ### **Why Recall Opt XGB Wins?**
    In medical screening, our goal is to capture as many 'True Positive' risk profiles as possible. While the Elite Ensemble is more accurate overall, the Recall-Optimized model ensures more people are flagged for clinical HbA1c testing, directly supporting early intervention.
    """)
    st.caption("Intelligence Analysis built with 5Base Models + Stacked Ensemble Classifiers.")

# --- TAB 1: ASSESSMENT ---
with tab1:
    st.title("🩺 Clinical Risk Assessment")
    st.write("Complete the profile below for a real-time risk evaluation.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Demographics")
        age_map = {
            "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
            "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
            "70-74": 11, "75-79": 12, "80+": 13
        }
        age = st.selectbox("Current Age Range", list(age_map.keys()))
        sex = st.radio("Biological Gender", ["Female", "Male"], horizontal=True)
        
        st.subheader("Primary Metrics")
        bmi = st.number_input("Body Mass Index (BMI)", 10.0, 100.0, 25.0)
        # WHO Category feedback
        if bmi < 18.5: st.warning(f"Classification: Underweight")
        elif bmi < 25: st.success(f"Classification: Healthy Weight")
        elif bmi < 30: st.info(f"Classification: Overweight")
        else: st.error(f"Classification: Clinically Obese")
        
        gen_hlth_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
        gen_hlth = st.radio("Self-Assessed General Health", list(gen_hlth_map.keys()), horizontal=True)

    with col2:
        st.subheader("Clinical History")
        high_bp = st.checkbox("Diagnosed Hypertension (High BP)")
        high_chol = st.checkbox("Diagnosed Dyslipidemia (High Chol)")
        chol_check = st.checkbox("Cholesterol Screening (Past 5 Years)", value=True)
        heart_disease = st.checkbox("History of Cardiac Events (Heart Disease)")
        stroke = st.checkbox("History of Cerebrovascular Events (Stroke)")
        
        st.subheader("Lifestyle Factors")
        smoker = st.checkbox("Smoked 100+ Cigarettes in Lifetime")
        phys_active = st.checkbox("Regular Physical Activity", value=True)
        fruits = st.checkbox("Consume Fruits Daily", value=True)
        veggies = st.checkbox("Consume Veggies Daily", value=True)
        hvy_alcohol = st.checkbox("Heavy Alcohol Intake")
        
        diff_walk = st.checkbox("Difficulty with Mobility (Climbing/Walking)")
        ment_hlth = st.slider("Days of Poor Mental Health (Monthly)", 0, 30, 0)
        phys_hlth = st.slider("Days of Poor Physical Health (Monthly)", 0, 30, 0)
        
        # Required for features but less prominent
        income_map = {"<$10k": 1, "$10k-15k": 2, "$15k-20k": 3, "$20k-25k": 4, "$25k-35k": 5, "$35k-50k": 6, "$50k-75k": 7, "$75k+": 8}
        income = 5 # Defaulting
        edu_map = {"No HS": 1, "Elem": 2, "Some HS": 3, "HS Grad": 4, "Some College": 5, "Coll Grad": 6}
        edu = 4 # Defaulting
        healthcare = 1 # Defaulting
        doc_cost = 0 # Defaulting

    # Prepare Data
    input_data = {
        'HighBP': int(high_bp), 'HighChol': int(high_chol), 'CholCheck': int(chol_check),
        'BMI': bmi, 'Smoker': int(smoker), 'Stroke': int(stroke),
        'HeartDiseaseorAttack': int(heart_disease), 'PhysActivity': int(phys_active),
        'Fruits': int(fruits), 'Veggies': int(veggies), 'HvyAlcoholConsump': int(hvy_alcohol),
        'AnyHealthcare': healthcare, 'NoDocbcCost': doc_cost, 
        'GenHlth': gen_hlth_map[gen_hlth], 'MentHlth': float(ment_hlth), 'PhysHlth': float(phys_hlth),
        'DiffWalk': int(diff_walk), 'Sex': 1 if sex == "Male" else 0, 'Age': age_map[age],
        'Education': edu, 'Income': income
    }
    input_data['BMI_OBESE'] = 1 if bmi >= 30 else 0
    input_data['HIGH_RISK_COMBO'] = 1 if (high_bp and high_chol) else 0
    phys_hlth_flag = 1 if phys_hlth > 14 else 0
    input_data['POOR_HEALTH_SCORE'] = input_data['GenHlth'] + input_data['DiffWalk'] + phys_hlth_flag
    
    feature_cols = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income', 'BMI_OBESE', 'HIGH_RISK_COMBO', 'POOR_HEALTH_SCORE']
    input_df = pd.DataFrame([input_data])[feature_cols]
    input_scaled = scaler.transform(input_df)
    
    st.divider()
    
    if st.button("RUN CLINICAL RISK ANALYSIS"):
        model = models[selected_model_name]
        prob = model.predict_proba(input_scaled)[0][1]
        
        res_col1, res_col2 = st.columns([1, 1.3])
        
        with res_col1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Probability", 'font': {'size': 24, 'color': '#ffffff'}},
                number = {'font': {'color': '#4cc9f0', 'size': 50}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "#ffffff"},
                    'bar': {'color': "#4361ee"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [0, 20], 'color': 'rgba(76, 201, 240, 0.2)'},
                        {'range': [20, 60], 'color': 'rgba(67, 97, 238, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(247, 37, 133, 0.2)'}],
                }))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_column_width=True)
            
        with res_col2:
            st.markdown("### Risk Interpretation")
            if prob < (threshold if selected_model_name == '⭐ Elite Ensemble (Max Accuracy)' else threshold): # Dynamic threshold logic if needed
                st.success("#### PASS: LOW CLINICAL RISK")
                message = "Your profile suggests low immediate risk. Continue regular checkups."
            elif prob < 0.6:
                st.warning("#### WARNING: ELEVATED RISK")
                message = "Moderate markers detected. We recommend clinical consultation for blood glucose testing."
            else:
                st.error("#### ALERT: HIGH CLINICAL RISK")
                message = "Significant risk factors identified. Consult a physician immediately for diagnostic screenings."
            
            st.write(message)
            
            st.markdown("#### Primary Stressors")
            drivers = []
            if high_bp: drivers.append("🔹 Hypertension (Strong clinical link)")
            if bmi >= 30: drivers.append("🔹 Class 1+ Obesity (Metabolic driver)")
            if gen_hlth_map[gen_hlth] >= 4: drivers.append("🔹 Self-Identified Poor General Health")
            if age_map[age] >= 8: drivers.append("🔹 Age Interaction (Slowing metabolism)")
            if not phys_active: drivers.append("🔹 Physical Inactivity")
            
            for d in drivers[:4]:
                st.markdown(f"<div class='driver-item'>{d}</div>", unsafe_allow_html=True)

# --- TAB 2: EXPLORER ---
with tab2:
    st.title("📊 Population Risk Insights")
    st.write("Visualizing the relationship between lifestyle and disease across 253k patient records.")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### Risk by BMI Category")
        img_bmi = path_utils.get_outputs_path("diabetes_rate_by_bmi.png")
        if os.path.exists(img_bmi): 
            st.image(img_bmi, use_column_width=True)
            st.info("""
            **Clinical Insight:** Obesity (BMI ≥ 30) is the single most significant modifiable driver. 
            Data shows a **3x increase** in risk compared to the 'Healthy Weight' category.
            """)
    with col_e2:
        st.markdown("#### Risk by Age Progression")
        img_age = path_utils.get_outputs_path("diabetes_rate_by_age.png")
        if os.path.exists(img_age): 
            st.image(img_age, use_column_width=True)
            st.info("""
            **Clinical Insight:** Vulnerability increases sharply after Age Category 7 (45+ years). 
            Risk doubles for every two age categories above 40.
            """)

    st.divider()
    col_e3, col_e4 = st.columns(2)
    with col_e3:
        st.markdown("#### Feature Correlation Matrix")
        img_corr = path_utils.get_outputs_path("correlation_heatmap.png")
        if os.path.exists(img_corr): 
            st.image(img_corr, use_column_width=True)
            st.info("**Key Drivers:** GenHlth, HighBP, BMI, and Age show the strongest positive correlation with current and pre-diabetic status.")
    with col_e4:
        st.markdown("#### Diabetic Clinical Median")
        img_means = path_utils.get_outputs_path("feature_means_comparison.png")
        if os.path.exists(img_means): 
            st.image(img_means, use_column_width=True)
            st.info("**Pattern:** Patients with diabetes significantly exhibit co-occurring Hypertension and High Cholesterol ('High Risk Combo').")

# --- TAB 3: CALIBRATION ---
with tab3:
    st.title("📈 Model Intelligence & Metrics")
    st.write("Evaluating the predictive validity of the selected clinical model.")
    
    st.subheader("Area Under Curve (AUC-ROC) Comparison")
    img_roc = path_utils.get_outputs_path("roc_curves_elite_comparison.png") # Updated for Elite comparison
    if os.path.exists(img_roc):
        st.image(img_roc, use_column_width=True)
        st.success("**Model Evolution:** The Elite Stacked Ensemble achieves over 86% Accuracy, outperforming baseline models by effectively blending XGB, LGBM, and CatBoost.")
    
    col_m1, col_m2 = st.columns([1.5, 1])
    with col_m1:
        st.subheader("Global Feature Importance (XGBoost)")
        img_imp = path_utils.get_outputs_path("feature_importance.png")
        if os.path.exists(img_imp): st.image(img_imp, use_column_width=True)
            
    with col_m2:
        st.subheader("Comparative Metrics (Elite Stack)")
        metrics_file = path_utils.get_outputs_path("performance_metrics_elite.csv") # Updated for Elite metrics
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            st.dataframe(metrics_df.style.background_gradient(cmap='Blues', subset=['Accuracy']))
        
        st.subheader("Confusion Matrix (Elite Champion)")
        img_cm = path_utils.get_outputs_path("confusion_matrix_elite.png") # Updated for Elite CM
        if os.path.exists(img_cm): 
            st.image(img_cm, use_column_width=True)
            st.info("**Clinical Utility:** The Elite Model focuses on overall predictive accuracy, identifying the majority of non-diabetic cases with higher precision than the Recall-tuned XGBoost.")

# --- TAB 4: DISCLAIMER ---
with tab4:
    st.title("⚖️ Legal & Clinical Disclaimer")
    st.warning("PLEASE READ CAREFULLY")
    st.markdown("""
    ### 1. EDUCATIONAL PURPOSE
    This application is designed as a **technical showcase** of machine learning capabilities in the healthcare domain. It is **NOT** a medical diagnostic tool and should not be used as a substitute for professional medical advice.
    
    ### 2. PREDICTIVE NATURE
    Machine Learning models predict based on patterns found in historical population data (CDC BRFSS 2015). A predicted probability is a statistical estimate, not a clinical diagnosis.
    
    ### 3. ACTIONABLE ADVICE
    If this tool flags you as "High Risk," it serves as a prompt for you to **consult a licensed physician** for blood tests such as HbA1c or Fasting Plasma Glucose.
    
    ### 4. DATA PRIVACY
    All data processed in this session is volatile and cleared upon page refresh. No clinical data is stored in any database.
    """)
    st.info("Dataset: CDC Diabetes Health Indicators | Model: Elite Stacked Ensemble")

# --- FOOTER ---
st.markdown("""
<br><hr>
<center>
    <p style='color: #a0a0a0;'>Diabetes Risk Prediction System | Built by <b>Divyanshi Singh</b></p>
    <a href='https://github.com/Divyanshi018572' target='_blank'><img src='https://img.icons8.com/fluent/32/000000/github.png' width='25'/></a> &nbsp;
    <a href='https://www.linkedin.com/in/divyanshi-singh-ds/' target='_blank'><img src='https://img.icons8.com/fluent/32/000000/linkedin.png' width='25'/></a>
    <p style='color: #606060; font-size: 0.8em;'>© 2026 Professional Risk Engine | Data Science Portfolio</p>
</center>
""", unsafe_allow_html=True)
