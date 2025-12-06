import streamlit as st
import numpy as np
import joblib
import os

# Load model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_diabetes_models.pkl")
model = joblib.load(model_path)

st.title("Diabetes Prediction Web App")
st.write("Enter patient details to predict diabetes.")

Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose", 0, 300, 120)
BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
Insulin = st.number_input("Insulin", 0, 900, 80)
BMI = st.number_input("BMI", 0.0, 70.0, 25.5)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
Age = st.number_input("Age", 1, 120, 25)

user_input = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                        Insulin, BMI, DiabetesPedigreeFunction, Age]])

if st.button("Predict"):
    prediction = model.predict(user_input)[0]
    if prediction == 1:
        st.error("⚠️ High Risk: Patient is Diabetic")
    else:
        st.success("✅ Low Risk: Patient is NOT Diabetic")


