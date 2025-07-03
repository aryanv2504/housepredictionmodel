# app.py

import streamlit as st
import numpy as np
import joblib

# Load trained model and preprocessing steps
model = joblib.load('house_price_model.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

# Page title
st.title("🏠 House Price Prediction (INR)")

st.write("Fill in the details below to estimate the selling price of a house in Rupees.")

# User inputs matching features used during training
overall_qual = st.slider("Overall Quality (1 = Poor, 10 = Excellent)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=7000, value=1500)
garage_cars = st.slider("Garage Capacity (Cars)", 0, 5, 2)
total_sf = st.number_input("Total Square Footage (Basement + 1st & 2nd Floor)", min_value=300, max_value=10000, value=2000)
house_age = st.number_input("House Age (Years)", min_value=0, max_value=150, value=20)
remod_age = st.number_input("Remodel Age (Years)", min_value=0, max_value=150, value=5)
is_remodeled = st.selectbox("Has the house been remodeled?", ["No", "Yes"])

# Convert Yes/No to numeric
is_remodeled_num = 1 if is_remodeled == "Yes" else 0

# When user clicks the button
if st.button("Predict Price"):
    # Prepare the input array in correct order
    input_array = np.array([[overall_qual, gr_liv_area, garage_cars, total_sf, house_age, remod_age, is_remodeled_num]])
    
    # Apply preprocessing
    input_imputed = imputer.transform(input_array)
    input_scaled = scaler.transform(input_imputed)
    
    # Get prediction
    price_pred = model.predict(input_scaled)[0]
    
    # Display result in INR
    st.success(f"Estimated House Price: ₹{price_pred:,.0f}")
