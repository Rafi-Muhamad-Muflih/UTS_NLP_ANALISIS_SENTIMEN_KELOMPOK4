import streamlit as st
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Sentimen Shopee 3D Dark", page_icon="🛒", layout="centered")

# 2. CUSTOM CSS (DARK THEME + BLACK TEXT IN CARDS)
st.markdown("""
<style>
    /* Background Utama Hitam */
    .stApp {
        background: #000000;
        color: #ffffff;
    }

    /* Judul Utama Glow */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(#EE4D2D, #FF7337);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    /* Kartu Input (Tetap Glass Dark) */
    .input-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    /* KARTU HASIL (Warna Terang, Tulisan Hitam) */
    .result-card {
        background: rgba(255, 255, 255, 0.9); /* Putih agar tulisan hitam terlihat */
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.8), 0 0 20px rgba(238, 77, 45, 0.2);
        color: #000000 !important; /* Paksa tulisan jadi hitam */
        text-align: center;
        transform: perspective(1000px) rotateX(2deg);
        transition: 0.3s;
    }
    
    /* Styling Teks di dalam Kartu Hasil */
    .result-card h3, .result-card p, .result-card span {
        color: #000000 !important;
    }

    .pos-text { color: #1b5e20 !important; font-size: 32px; font-weight: bold; }
    .neg-text { color: #b71c1c !important; font-size: 32px; font-weight: bold; }
    
    /* Tombol Shopee */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #EE4D2D 0%, #FF7337 100%);
        color: white !important;
        border: none;
        padding: 12px;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(238, 77, 45, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    path = "model_indobert_deploy"
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)

try:
    classifier = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# 4. PREPROCESSING
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# 5. UI UTAMA
st.markdown('<h1 class="main-title">🛒 Analisis Sentimen E-commerce Shopee</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; opacity:0.7; color:white;">Berbasis IndoBERT</p>', unsafe_allow_html=True)

# Container Input
st.markdown('<div class="input-card">', unsafe_allow_html=True)
input_user = st.text_area("✍️ Masukkan ulasan pelanggan:", height=120)
analyze_button = st.button("MULAI ANALISIS")
st.markdown('</div>', unsafe_allow_html=True)

# Logika Prediksi
if analyze_button:
    if input_user.strip():
        with st.spinner('Sedang menghitung...'):
            prediction = classifier(clean(input_user))[0]
            label = prediction['label']
            score = prediction['score'] * 100
            
            if label == 'LABEL_1':
                hasil, gaya_text = "POSITIF 😊", "pos-text"
            else:
                hasil, gaya_text = "NEGATIF 😡", "neg-text"
            
            # Kartu Hasil Prediksi dengan Tulisan Hitam
            st.markdown(f"""
            <div class="result-card">
                <h3 style="margin-bottom:10px;">HASIL ANALISIS</h3>
                <hr style="border: 0.5px solid rgba(0,0,0,0.1);">
                <p class="{gaya_text}">{hasil}</p>
                <p style="font-size: 18px;">Tingkat Keyakinan: <b>{score:.2f}%</b></p>
                <div style="margin-top:15px; font-size:12px; opacity:0.6;">
                    Model: IndoBERT-base-p2 | Kelompok 4
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Masukkan ulasan terlebih dahulu.")

st.markdown('<br><p style="text-align:center; opacity:0.2; font-size:12px; color:white;">UAS KELOMPOK 4</p>', unsafe_allow_html=True)