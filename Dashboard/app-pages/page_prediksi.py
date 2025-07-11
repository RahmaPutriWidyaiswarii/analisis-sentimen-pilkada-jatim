import streamlit as st
import re
import nltk
#nltk.download('stopwords') # pastikan sudah mendownload nltk
import time
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import plotly.express as px
import pandas as pd

# ----------------------------------------------------------------------------
# Data Processing Code 
# ----------------------------------------------------------------------------

# Ambil kamus dan model dari session_state
if "norm_dict" not in st.session_state or "vectorizer" not in st.session_state  or "fselector" not in st.session_state or "model" not in st.session_state:
    st.error("Data belum dimuat! Silakan jalankan sentimen_app.py terlebih dahulu.")
else:
    norm_dict = st.session_state.norm_dict
    vectorizer = st.session_state.vectorizer
    fselector = st.session_state.fselector
    model = st.session_state.model

# Palet warna berdasarkan kelas sentimen
custom_colors = {'Negatif': '#EB5353', 'Netral': '#F5971D', 'Positif': '#36AE7C'}

# Peta label numerik ke string
label_mapping = {2: "Negatif", 0: "Netral", 1: "Positif"}

# Stopwords kombinasi
more_stop_words = ["loh", "lah", "dong"]
combined_stopwords = set(stopwords.words('indonesian') +
                         StopWordRemoverFactory().get_stop_words() +
                         more_stop_words)
stopword_dictionary = ArrayDictionary(list(combined_stopwords))
stop_words_remover = StopWordRemover(stopword_dictionary)

# Stemmer
stemmer = StemmerFactory().create_stemmer()

def preprocess_tweet(text):
    # 1. Cleaning
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'(?<=\w)\.(?=\w)', ' ', text)
    text = text.replace('\xa0', ' ')
    text = re.sub(r'[@#]\w+|RT[\s]+', ' ', text)
    text = re.sub(r'[0-9]', ' ', text)
    text = re.sub(r'[^A-Za-z ]', ' ', text)
    text = re.sub(r'[\n\r]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Case folding
    text = text.lower()

    # 3. Normalisasi kata
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in norm_dict.keys()) + r')\b')
    text = pattern.sub(lambda m: norm_dict[m.group(0)], text)

    # 4. Tokenisasi
    tokens = text.split()

    # 5. Stopword removal
    filtered_text = stop_words_remover.remove(' '.join(tokens)).split()

    # 6. Stemming
    stemmed = [stemmer.stem(word) for word in filtered_text]

    # 7. Hapus kata satu huruf
    final_tokens = [word for word in stemmed if len(word) > 1]

    # 8. Gabungkan kembali menjadi satu string
    final_text = ' '.join(final_tokens)

    # 9. Validasi hasil kosong
    return final_text if final_text.strip() else None

# ----------------------------------------------------------------------------
# Streamlit UI Code 
# ----------------------------------------------------------------------------

# --- Judul & Deskripsi ---
st.markdown("<h1 style='text-align:center;'>Prediksi Sentimen</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Silakan ketik pendapat Anda mengenai Pilkada Jatim 2024, dan sistem akan menganalisis sentimennya.</p>", unsafe_allow_html=True)

# --- Input chat-style ---
user_input = st.chat_input("Tulis pendapat Anda...")

# --- Jika ada input ---
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    clean_input = preprocess_tweet(user_input)

    with st.spinner("Memproses prediksi, mohon tunggu..."):
        time.sleep(1.2)

        vectorized_input = vectorizer.transform([clean_input])
        selected_vector = fselector.transform(vectorized_input)
        predicted_label_num = model.predict(selected_vector)[0]
        predicted_label = label_mapping[predicted_label_num]

        # Confidence score
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(selected_vector)[0]
            class_labels = model.classes_
            confidence_scores = {
                label_mapping[int(lbl)]: round(prob * 100, 2)
                for lbl, prob in zip(class_labels, probas)
            }
        else:
            confidence_scores = {predicted_label: 100.0}

        # Urutkan skor confidence
        sorted_scores = dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True))

        # Emoji dan warna
        emoji = {"Positif": "😊", "Negatif": "😠", "Netral": "😐"}
        color = custom_colors.get(predicted_label, "#000000")
        label_with_style = f"<span style='color: {color}; font-weight: bold;'>{predicted_label}</span>"

        # Format confidence score sebagai teks
        score_lines = ""

        # Paragraf formal sebagai balasan bot
        explanation = f"""<p style="text-align: justify;">Berdasarkan analisis yang dilakukan terhadap pendapat Anda, sentimen yang terdeteksi adalah {label_with_style} {emoji.get(predicted_label, '')}.</br>Model memiliki tingkat keyakinan tertentu dalam mengambil keputusan ini, sebagaimana divisualisasikan dalam diagram di bawah ini.</p>"""

        # Siapkan data untuk chart
        df_conf = pd.DataFrame({
            "Sentimen": list(sorted_scores.keys()),
            "Skor (%)": list(sorted_scores.values())
        })

        # Warna berdasarkan custom_colors
        color_map = custom_colors

        # Cari sentimen dengan skor tertinggi untuk explode
        max_index = df_conf["Skor (%)"].idxmax()
        pull_values = [0.1 if i == max_index else 0 for i in range(len(df_conf))]

        # Donut chart
        fig_donut = px.pie(
            df_conf,
            names="Sentimen",
            values="Skor (%)",
            hole=0.5,
            color="Sentimen",
            color_discrete_map=color_map
        )

        fig_donut.update_traces(
            pull=pull_values,
            textinfo='percent+label',
            textfont=dict(
                size=14,        # ukuran font
                color='white',  # warna tulisan
                family='Arial', # jenis font
            ),
            textfont_weight='bold'  # <- meskipun tidak didukung langsung, bisa simulasikan pakai family bold
        )

        fig_donut.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            showlegend=False,
        )

        # Tampilkan balasan bot
        with st.chat_message("assistant"):
            st.markdown(explanation, unsafe_allow_html=True)
            st.plotly_chart(fig_donut, use_container_width=True)