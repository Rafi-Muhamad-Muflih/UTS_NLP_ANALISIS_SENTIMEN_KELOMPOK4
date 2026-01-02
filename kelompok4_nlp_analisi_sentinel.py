import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =============================================================================
# 1. KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Analisis Sentimen Shopee",
    page_icon="🛒",
    layout="centered"
)

# =============================================================================
# 2. CUSTOM CSS (3D STYLE)
# =============================================================================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #f5f7fa, #e4ecf5);
}

.card {
    background: white;
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    text-align: center;
}

.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 25px 45px rgba(0,0,0,0.25);
}

.title-box {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    padding: 25px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.sentiment-positive {
    color: #1b5e20;
    font-weight: bold;
    font-size: 22px;
}

.sentiment-neutral {
    color: #f57f17;
    font-weight: bold;
    font-size: 22px;
}

.sentiment-negative {
    color: #b71c1c;
    font-weight: bold;
    font-size: 22px;
}

.sentiment-waiting {
    color: #9e9e9e;
    font-style: italic;
    font-size: 18px;
}

.footer {
    text-align: center;
    opacity: 0.6;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. DOWNLOAD RESOURCE NLTK
# =============================================================================
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =============================================================================
# 4. PREPROCESSING
# =============================================================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

list_stopwords = set(stopwords.words('indonesian'))

kata_penting = {
    'tidak','bukan','jangan','tak','belum','kurang',
    'gak','ga','nggak','kecewa','jelek','buruk','parah',
    'lama','lemot','susah','lumayan','biasa','cukup',
    'standar','tapi','agak','sedikit','saja'
}

list_stopwords = list_stopwords - kata_penting

def text_preprocessing(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    filtered_words = [w for w in words if w not in list_stopwords]
    return " ".join(filtered_words)

# =============================================================================
# 5. LOAD MODEL
# =============================================================================
@st.cache_resource
def load_assets():
    # Pastikan file pkl berada di folder yang sama dengan script
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model_nb.pkl', 'rb') as f:
        model_nb = pickle.load(f)
    with open('model_svm.pkl', 'rb') as f:
        model_svm = pickle.load(f)
    return vectorizer, model_nb, model_svm

# Eksekusi load model
try:
    vectorizer, model_nb, model_svm = load_assets()
except FileNotFoundError:
    st.error("File model (.pkl) tidak ditemukan. Pastikan file model ada di direktori yang sama.")
    st.stop()

# =============================================================================
# 6. HEADER
# =============================================================================
st.markdown("""
<div class="title-box">
    <h1>🛒 Analisis Sentimen Shopee</h1>
    <p>Dashboard Perbandingan Model Naive Bayes & SVM</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 7. INPUT CARD
# =============================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
input_text = st.text_area(
    "✍️ Masukkan Ulasan Pelanggan:",
    height=130,
    placeholder="Contoh: Barangnya lumayan bagus tapi pengirimannya lama"
)
analyze_button = st.button("🔍 Analisis Sentimen")
st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 8. DEFAULT CARD (TAMPILAN AWAL)
# =============================================================================
# Mengisi card kosong dengan status menunggu
label_nb, css_nb = ("Menunggu input ulasan...", "sentiment-waiting")
label_svm, css_svm = ("Menunggu input ulasan...", "sentiment-waiting")

# =============================================================================
# 9. PREDIKSI (DIJALANKAN SAAT TOMBOL DIKLIK)
# =============================================================================
if analyze_button:
    if input_text.strip() != "":
        with st.spinner('Sedang menganalisis...'):
            clean_text = text_preprocessing(input_text)
            vector = vectorizer.transform([clean_text])

            pred_nb = model_nb.predict(vector)[0]
            pred_svm = model_svm.predict(vector)[0]

            # Mapping label hasil prediksi
            label_map = {
                0: ("NEGATIF 😡", "sentiment-negative"),
                1: ("NETRAL 😐", "sentiment-neutral"),
                2: ("POSITIF 😊", "sentiment-positive")
            }

            label_nb, css_nb = label_map[pred_nb]
            label_svm, css_svm = label_map[pred_svm]
    else:
        st.warning("Silakan masukkan teks ulasan terlebih dahulu!")

# =============================================================================
# 10. HASIL PERBANDINGAN
# =============================================================================
st.markdown("## 📊 Perbandingan Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧮 Naive Bayes")
    st.markdown(f'<p class="{css_nb}">{label_nb}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ SVM")
    st.markdown(f'<p class="{css_svm}">{label_svm}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 11. FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    Kelompok 4 | UAS NLP – Analisis Sentimen Shopee
</div>
""", unsafe_allow_html=True)