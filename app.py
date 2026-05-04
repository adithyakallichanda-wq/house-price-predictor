import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Predictor")
st.markdown("### Enter property details below 👇")

sqft = st.number_input("Square Footage", value=1500.0)
bed = st.number_input("Bedrooms", value=3)
bath = st.number_input("Bathrooms", value=2)
year = st.number_input("Year Built", value=2005)
lot = st.number_input("Lot Size", value=2.5)
garage = st.number_input("Garage Size", value=1)
neigh = st.number_input("Neighborhood Quality (1-10)", value=7)

if st.button("Predict Price"):
    
    if year < 1900 or year > 2025:
        st.error("Enter valid year (1900–2025)")
    elif lot > 10:
        st.error("Lot size too large")
    else:
        data = np.array([[sqft, bed, bath, year, lot, garage, neigh]])
        data = scaler.transform(data)
        pred = model.predict(data)[0]

        if pred > 1e7:
            st.success(f"Estimated Price: ₹{pred/1e7:.2f} Cr")
        elif pred > 1e5:
            st.success(f"Estimated Price: ₹{pred/1e5:.2f} Lakh")
        else:
            st.success(f"Estimated Price: ₹{pred:,.2f}")