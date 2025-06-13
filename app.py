import streamlit as st
import tensorflow
from prediction import predict_mental_health

st.set_page_config(page_title="NervEase - Deteksi Awal Kesehatan Mental", layout="centered")

st.markdown("<h1 style='text-align: center;'>🧠 NervEase</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>"
            "Selamat datang di <b>NervEase</b> – platform skrining awal kesehatan mental berbasis AI.<br>"
            "Tuliskan curhatan atau deskripsi singkat tentang perasaan dan kondisi mental Anda saat ini.<br>"
            "Sistem akan memprediksi kemungkinan gangguan mental dan memberikan rekomendasi awal penanganan."
            "</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📝 Masukkan Curhatan Anda")
user_input = st.text_area("", placeholder="Contoh: Saya merasa cemas terus menerus, tidak bisa tidur nyenyak...", height=160)

if st.button("🔍 Prediksi Sekarang"):
    if not user_input.strip():
        st.warning("⚠️ Mohon isi curhatan terlebih dahulu.")
    else:
        label, rekomendasi = predict_mental_health(user_input)
        st.markdown("---")
        st.subheader("🧾 Hasil Analisis")
        st.markdown("**Prediksi Jenis Gangguan Mental:**")
        st.success(label)
        st.markdown("**Rekomendasi Penanganan:**")
        st.info(rekomendasi)

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>"
            "NervEase © 2025 – Dibuat oleh Tim NervEase Laskar AI untuk Kesehatan Mental Indonesia"
            "</p>", unsafe_allow_html=True)
