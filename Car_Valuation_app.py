import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained models
linear_model = joblib.load("linear_model.pkl")
lasso_model = joblib.load("lasso_model.pkl")

# Function to encode user inputs
def encode_input(year, km_driven, fuel, seller_type, transmission, owner):
    fuel_map = {'Diesel': 0, 'Petrol': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
    seller_map = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}
    trans_map = {'Manual': 0, 'Automatic': 1}
    owner_map = {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}

    return np.array([[year, km_driven,
                      fuel_map[fuel], seller_map[seller_type],
                      trans_map[transmission], owner_map[owner]]])

# Streamlit UI
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Prediction App")
st.write("Enter the car details below to get price predictions using trained machine learning models.")

# Input fields
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)
km_driven = st.number_input("Kms Driven", value=30000)
fuel = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

# Predict
if st.button("Predict Selling Price"):
    input_array = encode_input(year, km_driven, fuel, seller_type, transmission, owner)
    feature_names = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
    input_data = pd.DataFrame(input_array, columns=feature_names)

    try:
        linear_pred = linear_model.predict(input_data)[0]
        lasso_pred = lasso_model.predict(input_data)[0]

        st.success(f"💡 Linear Regression Prediction: ₹ {linear_pred:,.2f}")
        st.success(f"💡 Lasso Regression Prediction: ₹ {lasso_pred:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
