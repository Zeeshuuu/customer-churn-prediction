# ============================================
# TELCO CUSTOMER CHURN PREDICTION DASHBOARD (XAI EDITION)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# -----------------------------
# 1️⃣ PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown(
    "This explainable AI dashboard predicts **customer churn probability** and explains why the model made that decision using **SHAP**."
)

# -----------------------------
# 2️⃣ LOAD MODEL, SCALER, FEATURES
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_model_gradient_boosting.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features

model, scaler, model_features = load_model()
st.success("✅ Model, Scaler, and Features loaded successfully!")

# -----------------------------
# 3️⃣ USER INPUT SECTION
# -----------------------------
st.header("📋 Input Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 2000.0)

# -----------------------------
# 4️⃣ PROCESS INPUT
# -----------------------------
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# One-hot encode categorical features
df_encoded = pd.get_dummies(input_data)

# Align with model feature columns
df_encoded = df_encoded.reindex(columns=model_features, fill_value=0)

# Scale numerical data
scaled_data = scaler.transform(df_encoded)

# -----------------------------
# 5️⃣ PREDICTION
# -----------------------------
if st.button("🔍 Predict Churn"):
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("🔮 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The model predicts: **Customer WILL Churn** (Risk: {probability:.2f})")
    else:
        st.success(f"💚 The model predicts: **Customer Will Stay** (Risk: {probability:.2f})")

    # -----------------------------
    # 6️⃣ SHAP EXPLANATION (LOCAL)
    # -----------------------------
    st.header("💧 SHAP Explanation (Feature Impact on This Prediction)")

    explainer = shap.Explainer(model)
    shap_values = explainer(df_encoded)

    sample_shap = shap_values[0]

    # Fix base_values shape if needed
    if isinstance(sample_shap.base_values, np.ndarray):
        sample_shap.base_values = float(np.mean(sample_shap.base_values))

    # Waterfall plot
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(sample_shap, max_display=10, show=False)
    st.pyplot(fig)

# -----------------------------
# 7️⃣ GLOBAL FEATURE IMPORTANCE
# -----------------------------
st.header("📊 Global Feature Importance (Model Insights)")

try:
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': model_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(15), palette="viridis", ax=ax2)
    ax2.set_title("Top 15 Important Features Influencing Churn")
    st.pyplot(fig2)

except Exception as e:
    st.warning(f"⚠️ Could not compute feature importances: {e}")
