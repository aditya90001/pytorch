import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load model and columns
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_diabetes_models.pkl")
columns_path = os.path.join(current_dir, "columns.pkl")

model = joblib.load(model_path)
columns = joblib.load(columns_path)  # list of feature names used in training

st.title("Diabetes Prediction Web App")
st.write("Enter patient details to predict diabetes. Use sliders to adjust values.")

# Healthy default values
default_values = {
    "Pregnancies": 1,
    "Glucose": 85,
    "BloodPressure": 72,
    "SkinThickness": 18,
    "Insulin": 90,
    "BMI": 23.5,
    "DiabetesPedigreeFunction": 0.3,
    "Age": 28
}

# Sliders for inputs
Pregnancies = st.slider("Pregnancies", 0, 20, default_values["Pregnancies"])
Glucose = st.slider("Glucose", 0, 300, default_values["Glucose"])
BloodPressure = st.slider("Blood Pressure", 0, 200, default_values["BloodPressure"])
SkinThickness = st.slider("Skin Thickness", 0, 100, default_values["SkinThickness"])
Insulin = st.slider("Insulin", 0, 900, default_values["Insulin"])
BMI = st.slider("BMI", 0.0, 70.0, default_values["BMI"])
DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.0, 3.0, default_values["DiabetesPedigreeFunction"])
Age = st.slider("Age", 1, 120, default_values["Age"])

if st.button("Predict"):
    # Convert user input to DataFrame with correct column names
    user_input = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    user_input_df = pd.DataFrame(user_input, columns=columns)

    # Predict
    prediction = model.predict(user_input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk: Patient is Diabetic")
    else:
        st.success("✅ Low Risk: Patient is NOT Diabetic")
