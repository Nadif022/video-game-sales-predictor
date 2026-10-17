import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('model_game_sales.pkl')

st.title("Prediksi Penjualan Video Game")

# Contoh input sederhana (sesuaikan dengan fitur model Anda)
platform = st.selectbox("Pilih Platform", ["PS4", "XOne", "Switch", "PC"])
genre = st.selectbox("Pilih Genre", ["Action", "Shooter", "Sports", "RPG"])

if st.button("Prediksi"):
    # Buat dataframe input
    input_data = pd.DataFrame([[platform, genre]], columns=['Platform', 'Genre'])
    prediction = model.predict(input_data)
    st.success(f"Estimasi Penjualan: {prediction[0]:.2f} Juta Unit")