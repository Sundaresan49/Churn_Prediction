import streamlit as st
import joblib
import numpy as np

# Corrected file path with proper string formatting
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title("Churn Prediction")
st.divider()
st.write("Please enter the values and hit the predict button for getting prediction")
st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Select Gender", ["Male", "Female"])
tenure = st.number_input("Enter tenure", min_value=0, max_value=130, value=10)
monthlycharges = st.number_input("Enter monthly charges", min_value=30.0, max_value=150.0)
st.divider()

predictbutton = st.button("Predict")
if predictbutton:
    # Corrected the gender selection logic
    gender_selected = 1 if gender == "Female" else 0
    x = [age, gender_selected, tenure, monthlycharges]
    
    # Corrected variable name from x1 to x
    x_array = scaler.transform([x])
    
    # Corrected prediction logic
    prediction = model.predict(x_array)[0]
    predicted = "churn" if prediction == 1 else "not churn"
    st.write(f"predicted: {predicted}")
else:
    st.write("please click on the predict button to get predictions")
