import streamlit as st
from PIL import Image

# Konfigurasi halaman
st.set_page_config(page_title="NervEase - Deteksi Awal Kesehatan Mental", layout="centered")

# Header dengan emoji dan highlight
st.markdown("<h1 style='text-align: center;'>🧠 NervEase</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>"
            "Selamat datang di <b>NervEase</b> – platform skrining awal kesehatan mental berbasis AI.<br>"
            "Tuliskan curhatan atau deskripsi singkat tentang perasaan dan kondisi mental Anda saat ini.<br>"
            "Sistem akan memprediksi kemungkinan gangguan mental dan memberikan rekomendasi awal penanganan."
            "</p>", unsafe_allow_html=True)

# Garis pemisah
st.markdown("---")

# Input dari pengguna
st.markdown("### 📝 Masukkan Curhatan Anda")
user_input = st.text_area("", placeholder="Contoh: Akhir-akhir ini saya merasa tidak bersemangat, sulit tidur, dan mudah marah...", height=160)

# Tombol prediksi
if st.button("🔍 Prediksi Sekarang"):
    if not user_input.strip():
        st.warning("⚠️ Mohon isi curhatan terlebih dahulu sebelum memprediksi.")
    else:
        # Simulasi prediksi model (placeholder)
        predicted_label = "Anxiety"
        rekomendasi = (
              "Cobalah atur waktu istirahat dan aktivitasmu secara seimbang."
              "Lakukan hal-hal yang kamu sukai, seperti mendengarkan musik, journaling, atau olahraga ringan.\n\n"
              "Latihan pernapasan dan mindfulness bisa membantu meredakan pikiran yang terlalu penuh.\n\n"
              "Kalau rasa cemas mulai mengganggu aktivitas harianmu, jangan ragu untuk ngobrol dengan psikolog atau konselor. "
        )

        # Hasil prediksi
        st.markdown("---")
        st.subheader(" Hasil Analisis ")

        with st.container():
            st.markdown(f"** Prediksi Jenis Gangguan Mental:**")
            st.success(predicted_label)

            st.markdown("** Rekomendasi Penanganan:**")
            st.info(rekomendasi)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>"
            "NervEase © 2025 – Dibuat oleh Tim NervEase Laskar AI untuk Kesehatan Mental Indonesia"
            "</p>", unsafe_allow_html=True)
