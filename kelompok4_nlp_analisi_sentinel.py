import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =============================================================================
# 1. KONFIGURASI HALAMAN & RESOURCE
# =============================================================================
st.set_page_config(page_title="Analisis Sentimen Shopee", page_icon="🛒")

# Download resource NLTK (Hanya sekali jalan, aman ditaruh di sini)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# =============================================================================
# 2. FUNGSI PREPROCESSING (WAJIB SAMA PERSIS DENGAN TRAINING)
# =============================================================================
# Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Konfigurasi Stopwords
list_stopwords = set(stopwords.words('indonesian'))

# DAFTAR KATA PENTING (Sama persis dengan script training terakhir)
kata_penting = {
    # Kata Negatif
    'tidak', 'bukan', 'jangan', 'tak', 'belum', 'kurang', 'gak', 'ga', 'nggak',
    'kecewa', 'jelek', 'buruk', 'parah', 'lama', 'lemot', 'susah', 'bangsat', 'hancur',
    
    # Kata Netral / Penyeimbang
    'lumayan', 'biasa', 'cukup', 'standar', 'tapi', 'agak', 'sedikit', 'baja', 'saja', 
    'kadang', 'bingung', 'standart', 'not bad', 'kurang lebih'
}

# Hapus kata penting dari daftar stopwords
list_stopwords = list_stopwords - kata_penting

def text_preprocessing(text):
    # a. Case Folding
    text = str(text).lower()

    # b. Cleaning
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # c. Tokenizing & Stopword Removal
    words = text.split()
    filtered_words = [word for word in words if word not in list_stopwords]

    # d. Gabung kembali
    processed_text = " ".join(filtered_words)
    return processed_text

# =============================================================================
# 3. LOAD MODEL & VECTORIZER
# =============================================================================
@st.cache_resource # Agar tidak load ulang setiap kali klik tombol
def load_assets():
    try:
        with open('model_sentiment.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        with open('vectorizer_tfidf.pkl', 'rb') as vec_file:
            loaded_vec = pickle.load(vec_file)
        return loaded_model, loaded_vec
    except FileNotFoundError:
        return None, None

model, vectorizer = load_assets()

# =============================================================================
# 4. TAMPILAN ANTARMUKA (UI)
# =============================================================================
st.title("🛒 Analisis Sentimen Ulasan Shopee")
st.markdown("Aplikasi untuk mendeteksi sentimen: **Positif**, **Netral**, atau **Negatif**.")
st.markdown("---")

# Cek apakah model berhasil di-load
if model is None or vectorizer is None:
    st.error("⚠️ File 'model_sentiment.pkl' atau 'vectorizer_tfidf.pkl' tidak ditemukan! Pastikan file ada di folder yang sama.")
    st.stop()

# Input User
input_text = st.text_area("Masukkan Ulasan Pelanggan:", height=150, placeholder="Contoh: Barangnya lumayan bagus, tapi pengiriman agak lama...")

if st.button("🔍 Analisis Sentimen"):
    if input_text:
        with st.spinner('Sedang menganalisis...'):
            # 1. Preprocessing
            clean_text = text_preprocessing(input_text)
            
            # 2. Vectorization (Ubah ke angka)
            text_vector = vectorizer.transform([clean_text])
            
            # 3. Prediksi (Outputnya Angka: 0, 1, atau 2)
            prediksi_angka = model.predict(text_vector)[0]
            
            # 4. Logika Mapping Output (0=Negatif, 1=Netral, 2=Positif)
            st.markdown("### Hasil Analisis:")
            
            if prediksi_angka == 2: # Positif
                st.success(f"😊 **POSITIF**")
                st.write("Ulasan ini mengandung nada kepuasan atau pujian.")
                
            elif prediksi_angka == 1: # Netral
                st.warning(f"😐 **NETRAL**")
                st.write("Ulasan ini cenderung biasa saja, standar, atau memiliki sentimen campuran (bagus tapi ada kurangnya).")
                
            elif prediksi_angka == 0: # Negatif
                st.error(f"😡 **NEGATIF**")
                st.write("Ulasan ini mengandung keluhan, kekecewaan, atau kemarahan.")
                
            # (Opsional) Tampilkan teks bersih untuk debug
            with st.expander("Lihat Hasil Preprocessing (Debug)"):
                st.text(f"Original: {input_text}")
                st.text(f"Cleaned : {clean_text}")

    else:
        st.info("Silakan masukkan teks ulasan terlebih dahulu.")

# Footer
st.markdown("---")
st.caption("Dikembangkan oleh Kelompok 4 NLP")