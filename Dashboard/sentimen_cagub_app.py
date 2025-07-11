import pandas as pd
import streamlit as st
import pickle

# Konfigurasi awal Streamlit
st.set_page_config(page_title="Analisis Sentimen Pemilihan Calon Gubernur Jawa Timur 2024", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------------
# Data Processing Code 
# ----------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("Data/data_cagub_analisis.csv")
    
    name_mapping = {
        'Luluk': 'Luluk Nur Hamidah',
        'Khofifah': 'Khofifah Indar Parawansa',
        'Risma': 'Tri Rismaharini'
    }
    df['tokoh'] = df['tokoh'].map(name_mapping)
    
    return df

@st.cache_data
def load_kamus():
    df_kamus = pd.read_csv('Data/Kamus Normalisasi.csv', encoding='ISO-8859-1', header=None)
    df_kamus.columns = ['Alay', 'Meaning']
    norm_dict = dict(zip(df_kamus.Alay, df_kamus.Meaning))
    
    # Update manual untuk kata-kata khusus
    norm_dict.update({
        'p': 'perjuangan',
        'iniiii': 'ini',
        'bauuuuu': 'bau',
        'anake': 'anak',
        'ibuibu': 'ibu ibu',
        'nu': 'nahdlatul ulama',
        'pragib': 'prabowo gibran',
        'dahal': 'padahal',
        'lagiiiii': 'lagi',
        'kopipa': 'khofifah',
        'kopipah': 'khofifah',
        'khopipah': 'khofifah',
        'hopipah': 'khofifah'
    })

    return norm_dict


# Load model
@st.cache_resource 
def load_model():
    with open("Model/best_saved_tfidf_vectorizer.pkl", "rb") as f_vec, open("Model/best_saved_selector.pkl", "rb") as f_select, open("Model/best_saved_rf_model.pkl", "rb") as f_model:
        vectorizer = pickle.load(f_vec)
        fselector = pickle.load(f_select)
        model = pickle.load(f_model)
    return vectorizer, fselector, model

# Inisialisasi session_state untuk data dan kamus normalisasi
if "df_sentimen" not in st.session_state or "norm_dict" not in st.session_state:
    st.session_state.df_sentimen = load_data()
    st.session_state.norm_dict = load_kamus()

# Inisialisasi session_state untuk model
if "vectorizer" not in st.session_state or "model" not in st.session_state:
    vectorizer, fselector, model = load_model()
    st.session_state.vectorizer = vectorizer
    st.session_state.fselector = fselector
    st.session_state.model = model

# Inisialisasi session_state untuk filter
if "select_cagub" not in st.session_state:
    st.session_state.select_cagub = "Luluk Nur Hamidah"

if "label_sentimen" not in st.session_state:
    st.session_state.label_sentimen = "All"

if "top_number" not in st.session_state:
    st.session_state.top_number = 10

# Gunakan data dari session state, tanpa memuat ulang
df_sentimen = st.session_state.df_sentimen


# ----------------------------------------------------------------------------
# Streamlit UI Code 
# ----------------------------------------------------------------------------


# Page Setup
pages = {
    "Menu Navigasi": [
        st.Page(page="app-pages/page_dashboard.py", title="Dashboard", icon="📈", default=True),
        st.Page(page="app-pages/page_prediksi.py", title="Prediksi Sentimen", icon="💬")
    ]
}

# Navigation setup
pg = st.navigation(pages)

# Global filter
# Sidebar: Pilih filter
st.sidebar.subheader("🔍 Filters")

# Sidebar: Filter pilih cagub
list_cagub = ["Luluk Nur Hamidah", "Khofifah Indar Parawansa", "Tri Rismaharini"]
select_cagub = st.sidebar.selectbox("Calon Gubernur:", options=list_cagub, index=0)

# Sidebar: Filter label sentimen
sentimen_cat = ["All"] + sorted(df_sentimen["Sentimen"].unique())
label_sentimen = st.sidebar.selectbox("Jenis Sentimen:", options=sentimen_cat, index=0)

# Sidebar: Filter top data
top_number = st.sidebar.number_input("Jumlah Data Teratas", min_value=5, max_value=50, value=20)

# Update session_state jika ada perubahan & refresh halaman
if (
    select_cagub != st.session_state.select_cagub or
    label_sentimen != st.session_state.label_sentimen or
    top_number != st.session_state.top_number
):
    st.session_state.select_cagub = select_cagub
    st.session_state.label_sentimen = label_sentimen
    st.session_state.top_number = top_number
    st.rerun()  # Refresh agar filter berlaku

pg.run()
