import streamlit as st
import pandas as pd
import numpy as np
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

# Example input fields (you can expand these as per your dataset features)
age = st.number_input("Age", min_value=15, max_value=22, value=18)
studytime = st.number_input("Weekly Study Time (1-4)", min_value=1, max_value=4, value=2)
failures = st.number_input("Past Class Failures", min_value=0, max_value=4_
