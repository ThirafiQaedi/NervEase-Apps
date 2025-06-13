import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_input

# Load model LSTM
model = load_model("model_LSTM_fin.h5")

# Label kelas
label_mapping = {
    0: 'BPD',
    1: 'bipolar',
    2: 'depression',
    3: 'Anxiety',
    4: 'schizophrenia',
    5: 'mentalillness'
}

# Rekomendasi untuk setiap label
recommendations = {
    "BPD": "Lakukan terapi reguler, pelajari manajemen emosi, dan jaga hubungan interpersonal yang sehat.",
    "bipolar": "Jaga pola tidur dan aktivitas harian, dan konsultasikan dengan psikiater untuk pengobatan dan terapi.",
    "depression": "Coba aktivitas fisik ringan, jaga komunikasi sosial, dan pertimbangkan bantuan profesional.",
    "Anxiety": "Latihan pernapasan, mindfulness, dan hindari konsumsi kafein berlebihan. Konsultasi jika cemas berkepanjangan.",
    "schizophrenia": "Terapi dan pengobatan rutin sangat penting. Dukungan keluarga dan lingkungan aman sangat membantu.",
    "mentalillness": "Langkah awal seperti journaling, curhat, dan konsultasi psikolog bisa membantu mengidentifikasi masalah."
}

def predict_mental_health(text):
    processed = preprocess_input(text)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction, axis=1)[0]
    label = label_mapping[predicted_class]
    rekomendasi = recommendations[label]
    return label, rekomendasi
