---
title: DiaRisk-Advanced Detection Engine
emoji: 🩺
colorFrom: blue
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# 🛡️ DiaRisk-Advanced Detection Engine

**DiaRisk** is a professional-grade clinical diagnostic system designed to identify Type 2 Diabetes risk patterns by transforming 253k+ CDC real-world health records into actionable medical insights.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg?color=blue)](https://huggingface.co/spaces/YOUR_USERNAME/diarisk-detection-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](https://opensource.org/licenses/MIT)

## 🩺 Project Intelligence & Mission
The core challenge in diabetes screening is balancing **Sensitivity (Recall)** vs. **Accuracy**. Missing a high-risk patient is clinically dangerous. DiaRisk addresses this by providing two optimized model tiers:
1.  **Recall-Optimized XGBoost**: Designed for clinical safety, ensuring 79% of potential cases are flagged for screening.
2.  **Elite Stacked Ensemble**: An advanced booster blending XGBoost, LightGBM, and CatBoost for a peak **86.4% Accuracy**.

## 📊 Key Clinical Insights
Our exploratory analysis of the CDC BRFSS dataset revealed critical risk drivers:
*   **Obesity Impact**: Patients with BMI ≥ 30 show a **3x higher risk** of diabetes compared to those at a healthy weight.
*   **Age Progression**: Risk doubles for every two age categories above 40, with a sharp spike after age 45.
*   **General Health**: Self-assessed "Poor" or "Fair" health status is the strongest non-clinical predictor of current diabetic status.

## 🏆 Model Performance
| Model Tier | Accuracy | AUC-ROC | Recall (Sensitivity) | Focus |
|:---|:---|:---|:---|:---|
| **Elite Ensemble** | **86.4%** | **0.83** | 21% | Technical Precision |
| **XGBoost (Recall)** | 72.2% | **0.83** | **79.3%** | Clinical Safety |

## 🛠️ Technology Stack
- **Languages**: Python 3.10+
- **Boosters**: XGBoost, LightGBM, CatBoost
- **Ensemble**: Scikit-learn StackingClassifier
- **Dashboard**: Streamlit (Navy/Dark Clinical UI)
- **Visuals**: Plotly, Seaborn, Matplotlib

## 📂 Repository Structure
```text
├── data/               # Raw & Processed CDC Datasets
├── models/             # Serialized (.pkl) Elite Model Checkpoints
├── pipeline/           # End-to-end ML Scripts (EDA, Prep, Train, Eval)
├── outputs/            # Diagnostic Visualizations & Metrics
├── app.py              # Main Dashboard Entry Point
└── path_utils.py       # Global Path Management Utility
```

## 🏗️ Installation & Usage
1. Clone the repository: `git clone https://github.com/Divyanshi018572/diarisk-detection-engine.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch Dashboard: `streamlit run app.py`

---
**Build by Divyanshi Singh** | [GitHub](https://github.com/Divyanshi018572) | [LinkedIn](https://www.linkedin.com/in/divyanshi-singh-ds/)
