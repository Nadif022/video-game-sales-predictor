import joblib
import pandas as pd
import streamlit as st

# 1. Judul Aplikasi
st.title("🎮 Video Game Sales Prediction")
st.write(
    "Aplikasi web sederhana untuk memprediksi penjualan game menggunakan model"
    " Machine Learning."
)

# 2. Muat Model (.joblib)
@st.cache_resource
def load_model():
  return joblib.load("model.joblib")


model = load_model()

# 3. Form Input untuk Pengguna (Sesuaikan dengan fitur di dataset Anda)
st.subheader("Masukkan Detail Game:")
platform = st.selectbox(
    "Platform", ["PS4", "XOne", "PC", "Switch", "PS3"]
)  # Contoh
genre = st.selectbox(
    "Genre", ["Action", "Sports", "Shooter", "RPG", "Misc"]
)  # Contoh
critic_score = st.slider("Critic Score", 0.0, 100.0, 75.0)
user_score = st.slider("User Score", 0.0, 10.0, 7.0)

# 4. Tombol Prediksi
if st.button("Prediksi Sekarang"):
  # Bungkus input menjadi DataFrame
  input_data = pd.DataFrame({
      "Platform": [platform],
      "Genre": [genre],
      "Critic_Score": [critic_score],
      "User_Score": [user_score],
      # Tambahkan kolom lain sesuai dengan yang diterima model Anda
  })

  try:
    # Lakukan Prediksi
    prediction = model.predict(input_data)

    # Tampilkan Hasil
    st.success(f"Hasil Prediksi Penjualan: **{prediction[0]:.2f} Juta Unit**")

    # (OPSIONAL) Monitoring Sederhana: Cetak log di terminal/konsol cloud
    print(
        f"[LOG] Input: {input_data.to_dict()} | Prediksi: {prediction[0]}"
    )

  except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses data: {e}")
