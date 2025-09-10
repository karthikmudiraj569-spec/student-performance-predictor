import streamlit as st
import pandas as pd
import joblib
import os

# Paths for model and scaler
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Streamlit app
st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict the final grade (G3).")

# Example input fields (keep it simple for now)
age = st.number_input("Age", min_value=15, max_value=22, value=18)
studytime = st.number_input("Weekly Study Time (1-4)", min_value=1, max_value=4, value=2)
failures = st.number_input("Past Class Failures", min_value=0, max_value=4, value=0)
absences = st.number_input("Absences", min_value=0, max_value=100, value=5)
G1 = st.number_input("Grade Period 1 (0-20)", min_value=0, max_value=20, value=10)
G2 = st.number_input("Grade Period 2 (0-20)", min_value=0, max_value=20, value=12)

# Build dataframe for prediction
input_data = pd.DataFrame([{
    "age": age,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "G1": G1,
    "G2": G2
}])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Final Grade"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🎯 Predicted Final Grade (G3): {round(prediction, 2)}")

