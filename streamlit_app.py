import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

st.title("🎮 Video Game Sales Prediction")
st.write(
    "Aplikasi web interaktif untuk memprediksi penjualan video game menggunakan"
    " Machine Learning."
)


# 1. Load data dan latih model langsung di sini (Aman dari error versi Python/Joblib)
@st.cache_resource
def get_trained_model():
  # Baca dataset langsung (pastikan nama file CSV sesuai di GitHub)
  df = pd.read_csv("Raw Data.csv")

  # Lakukan pembersihan sederhana atau pilih fitur yang Anda gunakan
  # Sesuaikan kolom fitur (X) dan target (y) dengan project Anda di Jupyter Notebook
  # Contoh sederhana:
  df = df.dropna(subset=["Global_Sales", "Critic_Score", "User_Score"])

  X = df[["Critic_Score", "User_Score"]]  # Tambahkan fitur lain jika ada
  y = df["Global_Sales"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  model = GradientBoostingRegressor(random_state=42)
  model.fit(X_train, y_train)
  return model


try:
  model = get_trained_model()
  st.success("Model berhasil dimuat dan siap digunakan!")
except Exception as e:
  st.error(f"Gagal memuat data/model: {e}")

# 2. Form Input untuk Pengguna
st.subheader("Masukkan Parameter Game:")
critic_score = st.slider("Critic Score (0-100)", 0.0, 100.0, 75.0)
user_score = st.slider("User Score (0-10)", 0.0, 10.0, 7.0)

# 3. Tombol Prediksi
if st.button("Prediksi Penjualan"):
  input_data = pd.DataFrame(
      {
          "Critic_Score": [critic_score],
          "User_Score": [user_score],
      }
  )

  prediction = model.predict(input_data)
  st.markdown(f"### Hasil Prediksi Global Sales: **{prediction[0]:.2f} Juta Unit**")
