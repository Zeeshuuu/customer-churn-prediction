##  Customer Churn Prediction Dashboard

An interactive **AI-powered dashboard** built using **Streamlit** that predicts whether a telecom customer is likely to **churn (leave the service)** or **stay**.  
The project uses **machine learning models** trained on the **Telco Customer Churn Dataset** and includes **explainable AI visualizations (SHAP)** to interpret predictions.

#  Features

=> Predict customer churn using Gradient Boosting (best model)  
=> Compare performance of multiple models (Logistic Regression, Random Forest, XGBoost, Gradient Boosting)  
=> Interactive Streamlit dashboard for visualization  
=> SHAP explainability — see which features influence churn risk  
=> Data preprocessing, feature scaling, and model persistence  
=> Business insights from feature importance and customer trends

# Machine Learning Models Used

| Model Name           | Accuracy | ROC-AUC | F1-Score |
|----------------------|-----------|----------|-----------|
| Logistic Regression  | 0.7991    | 0.8403   | 0.5916    |
| Random Forest        | 0.7871    | 0.8251   | 0.5496    |
| XGBoost              | 0.7984    | 0.8366   | 0.5799    |
| **Gradient Boosting (Best)** | **0.8062** | **0.8416** | **0.5907** |

 **Best Model:** Gradient Boosting Classifier  
 Saved as: `best_model_gradient_boosting.pkl`

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

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Zeeshuuu/customer-churn-prediction.git
cd customer-churn-prediction
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Streamlit Dashboard
bash

streamlit run churn_dashboard.py
Then open the local URL (usually http://localhost:8501/) in your browser.

📊 Dataset Information
Dataset: Telco Customer Churn (from IBM Watson Analytics)

Rows: 7,000 customer records

Columns: Demographics, services, contracts, and billing details

Target Column: Churn (Yes / No)

The dataset helps telecom companies identify customers at risk of leaving and take proactive actions.

🧩 Model Training Process
Load and clean the Telco Churn dataset

Handle missing values in TotalCharges

Encode categorical variables using one-hot encoding

Scale numerical features with StandardScaler

Train multiple models:

Logistic Regression

Random Forest

XGBoost

Gradient Boosting

Compare accuracy, F1-score, and ROC-AUC

Save the best model and scaler for deployment

📈 SHAP Explainability
The dashboard integrates SHAP (SHapley Additive exPlanations) to:

Visualize how each feature influences churn probability

Provide local (individual) and global explanations

Help business teams understand why a customer is likely to churn

🌐 Business Impact
This project demonstrates how telecom or subscription-based businesses can:

Identify customers with high churn risk

Understand why they might leave

Take data-driven retention actions (discounts, better plans, etc.)

🧰 Technologies Used
Python 3.12

Streamlit — for interactive dashboard UI

Pandas & NumPy — for data processing

Scikit-learn — for ML model training

XGBoost — for gradient boosting model

SHAP — for explainability and model interpretation

Matplotlib & Seaborn — for visualizations

📬 Contact
Author: Zeeshan Memon
📧 Email: memonzeeshan2002@gmail.com
💼 GitHub: github.com/Zeeshuuu

