import re
import string
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt_tab')
nltk.download('stopwords')

# Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_json = f.read()  # baca sebagai string
tokenizer = tokenizer_from_json(tokenizer_json)


# Maksimal panjang input seperti saat training
MAXLEN = 100

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Hapus tanda baca
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Tokenisasi
    tokens = word_tokenize(text)
    # Stopword removal
    stop_words = set(stopwords.words('indonesian')) | set(stopwords.words('english'))
    custom_stopwords = {'iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', 'di', 'ga', 'ya', 'loh', 'kah'}
    stop_words |= custom_stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmed = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed)

def preprocess_input(text):
    cleaned = clean_text(text)
    sequences = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequences, maxlen=MAXLEN)
    return padded
