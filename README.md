# 💡 Customer Churn Prediction Dashboard

An **AI-powered interactive dashboard** built with **Streamlit** that predicts whether a telecom customer is likely to **churn (leave the service)** or **stay**.  
This project uses multiple **machine learning models** trained on the **Telco Customer Churn dataset** and includes **explainable AI visualizations (SHAP)** to interpret model predictions in a business-friendly way.


## ✨ Key Features

✅ Predict customer churn using **Gradient Boosting (best model)**  
✅ Compare performance of multiple ML models — Logistic Regression, Random Forest, XGBoost, Gradient Boosting  
✅ Interactive **Streamlit dashboard** for real-time visualization and prediction  
✅ **SHAP explainability** to understand which features impact churn risk  
✅ Complete **data preprocessing**, feature scaling, and model saving  
✅ Extract valuable **business insights** from feature importance and customer behavior trends  


## 🤖 Machine Learning Models & Performance

| Model Name             | Accuracy | ROC-AUC | F1-Score |
|-------------------------|----------|----------|-----------|
| Logistic Regression     | 0.7991   | 0.8403   | 0.5916    |
| Random Forest           | 0.7871   | 0.8251   | 0.5496    |
| XGBoost                 | 0.7984   | 0.8366   | 0.5799    |
| **Gradient Boosting (Best)** | **0.8062** | **0.8416** | **0.5907** |

🏆 **Best Model:** Gradient Boosting Classifier  
💾 **Saved as:** `best_model_gradient_boosting.pkl`


## 🗂 Project Structure

Customer-Churn-Prediction-AI
│
├── churn_dashboard.py # Streamlit dashboard (main app)
├── model_training.ipynb # Model training notebook
├── best_model_gradient_boosting.pkl # Trained Gradient Boosting model
├── scaler.pkl # StandardScaler for input data
├── model_features.pkl # Feature list used by model
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
├── requirements.txt # Project dependencies
└── README.md # Project documentation


## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Zeeshuuu/customer-churn-prediction.git
cd customer-churn-prediction
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Streamlit Dashboard
bash
Copy code
streamlit run churn_dashboard.py
Then open the local URL (usually http://localhost:8501/) in your browser.

📊 Dataset Information
Dataset: Telco Customer Churn (IBM Watson Analytics)
Rows: ~7,000 customer records
Columns: Demographics, services, contracts, and billing details
Target Column: Churn (Yes / No)

This dataset enables telecom companies to identify customers at risk of leaving and take proactive actions to reduce churn.

🧩 Model Training Workflow
Load and clean the Telco Churn dataset

Handle missing values in TotalCharges

Encode categorical variables with one-hot encoding

Scale numerical features using StandardScaler

Train multiple models:

Logistic Regression

Random Forest

XGBoost

Gradient Boosting

Compare models using Accuracy, F1-Score, and ROC-AUC

Save the best model, scaler, and feature list for deployment

📈 Explainable AI with SHAP
The dashboard integrates SHAP (SHapley Additive exPlanations) to make AI decisions transparent.

Understand how each feature affects churn probability

View both local (individual) and global (overall) model explanations

Empower business teams to make data-driven decisions with confidence

🌐 Business Impact
This project demonstrates how telecom or subscription-based businesses can:

Identify high-risk customers likely to churn

Understand why they might leave

Take proactive actions such as offering personalized discounts or better plans

Improve customer retention and reduce business losses

🧰 Tech Stack
🐍 Python 3.12

🎨 Streamlit — interactive dashboard

📊 Pandas & NumPy — data processing

🤖 Scikit-learn — ML model training

⚡ XGBoost — gradient boosting model

🔍 SHAP — model explainability

📈 Matplotlib & Seaborn — data visualization

📬 Contact
Author: Zeeshan Memon
📧 Email: memonzeeshan2002@gmail.com
💼 GitHub: github.com/Zeeshuuu
