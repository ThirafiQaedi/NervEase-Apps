import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# === CONFIG ===
MODEL_PATH = "model_LSTM_fin.keras"
TOKENIZER_PATH = "tokenizer.pkl"

# === KATEGORI DAN SARAN ===
class_names = [
    'Borderline Personality Disorder (BPD)',
    'Bipolar',
    'Depression',
    'Anxiety',
    'Schizophrenia',
    'Mentalillness'
]

rekomendasi_dict = {
    'Borderline Personality Disorder (BPD)': "Lakukan terapi reguler, pelajari manajemen emosi, dan jaga hubungan interpersonal yang sehat.",
    'Bipolar': "Jaga pola tidur dan aktivitas harian, dan konsultasikan dengan psikiater untuk pengobatan dan terapi.",
    'Depression': "Coba aktivitas fisik ringan, jaga komunikasi sosial, dan pertimbangkan bantuan profesional.",
    'Anxiety': "Latihan pernapasan, mindfulness, dan hindari konsumsi kafein berlebihan. Konsultasi jika cemas berkepanjangan.",
    'Schizophrenia': "Terapi dan pengobatan rutin sangat penting. Dukungan keluarga dan lingkungan aman sangat membantu.",
    'Mentalillness': "Langkah awal seperti journaling, curhat, dan konsultasi psikolog bisa membantu mengidentifikasi masalah."
}

# === LOAD MODEL DAN TOKENIZER ===
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# === FUNGSI PREDIKSI ===
def predict_text(text, model, tokenizer, maxlen=100):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    pred = model.predict(padded)[0]
    pred_index = np.argmax(pred)
    label = class_names[pred_index]
    conf = pred[pred_index] * 100
    return label, conf, rekomendasi_dict[label]

# === UI STREAMLIT ===
st.set_page_config(page_title="NervEase – Deteksi Awal Kesehatan Mental", layout="centered")

st.markdown("<h1 style='text-align: center;'>🧠 NervEase</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>"
            "Selamat datang di <b>NervEase</b> – platform skrining awal kesehatan mental berbasis AI.<br>"
            "Tuliskan curhatan atau deskripsi singkat tentang perasaan dan kondisi mental Anda saat ini.<br>"
            "Sistem akan memprediksi kemungkinan gangguan mental dan memberikan rekomendasi awal penanganan."
            "</p>", unsafe_allow_html=True)
st.markdown("---")

# Input
st.markdown("### 📝 Masukkan Curhatan Anda")
user_input = st.text_area("", placeholder="Contoh: Saya merasa tertekan, tidak punya semangat, dan tidak bisa tidur...", height=150)

# Prediksi
if st.button("🔍 Prediksi Sekarang"):
    if not user_input.strip():
        st.warning("⚠️ Masukkan tidak boleh kosong.")
    else:
        label, conf, saran = predict_text(user_input, model, tokenizer)
        st.markdown("---")
        st.subheader("🔎 Hasil Analisis")
        st.success(f"Jenis Gangguan Mental: **{label}** ({conf:.2f}%)")
        st.info(f"💡 Rekomendasi:\n{saran}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>"
            "NervEase © 2025 – Untuk Kesehatan Mental yang Lebih Baik"
            "</p>", unsafe_allow_html=True)