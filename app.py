import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os# app.py
import pickle

# Load model, scaler, and feature columns
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🎓 Student Performance Predictor")
st.markdown("Enter student details below to predict the final grade (G3):")

# Input form
with st.form("input_form"):
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.slider("Age", 15, 22, 17)
    studytime = st.selectbox("Study Time", [1, 2, 3, 4])
    failures = st.number_input("Past Failures", min_value=0, max_value=4, step=1)
    absences = st.slider("Absences", 0, 100, 4)
    higher = st.selectbox("Wants Higher Education?", ["yes", "no"])
    internet = st.selectbox("Has Internet Access?", ["yes", "no"])
    G1 = st.slider("1st Period Grade (G1)", 0, 20, 10)
    G2 = st.slider("2nd Period Grade (G2)", 0, 20, 10)
    
    submitted = st.form_submit_button("Predict Grade")

if submitted:
    # Prepare input data
    input_dict = {
        "sex": [sex],
        "age": [age],
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "higher": [higher],
        "internet": [internet],
        "G1": [G1],
        "G2": [G2]
    }

    input_df = pd.DataFrame(input_dict)

    # One-hot encode user input
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns
  feature_columns = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))

    input_encoded = input_encoded[feature_columns]  # Reorder columns

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"📘 Predicted Final Grade (G3): **{prediction:.2f}**")





