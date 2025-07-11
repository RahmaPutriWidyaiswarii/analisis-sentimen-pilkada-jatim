import re
import random
import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter


# ----------------------------------------------------------------------------
# Data Processing Code 
# ----------------------------------------------------------------------------

# Ambil data dari session_state
if "df_sentimen" not in st.session_state:
    st.error("Data belum dimuat! Silakan jalankan sentimen_app.py terlebih dahulu.")
else:
    df_sentimen = st.session_state.df_sentimen

# Ambil filter dari session_state
select_cagub = st.session_state.get("select_cagub")
label_sentimen = st.session_state.get("label_sentimen")
top_number = st.session_state.get("top_number")

# Warna untuk masing-masing kelas sentimen
custom_colors = {'Negatif': '#EB5353', 'Netral': '#F5971D', 'Positif': '#36AE7C'}

def hitung_jumlah(df):
    """
    Menghitung total data dan jumlah masing-masing kategori sentimen dari DataFrame.

    Args:
        df (pd.DataFrame): DataFrame yang mengandung kolom 'Sentimen'.

    Returns:
        tuple: Berisi total data, jumlah sentimen positif, netral, dan negatif
        Format: (total, positif_count, netral_count, negatif_count)
    """
    
    # Menghitung total jumlah data/baris dalam DataFrame
    total = df.shape[0]
    
    # Menghitung frekuensi masing-masing nilai sentimen
    sentimen_counts = df['Sentimen'].value_counts()
    
    # Mengambil jumlah sentimen positif, default 0 jika tidak ada
    positif_count = sentimen_counts.get('Positif', 0)
    
    # Mengambil jumlah sentimen netral, default 0 jika tidak ada
    netral_count = sentimen_counts.get('Netral', 0)
    
    # Mengambil jumlah sentimen negatif, default 0 jika tidak ada
    negatif_count = sentimen_counts.get('Negatif', 0)

    # Mengembalikan hasil perhitungan dalam bentuk tuple
    return total, positif_count, netral_count, negatif_count


def visualize_sentiment_distribution_by_tokoh(df):
    """
    Membuat visualisasi distribusi sentimen per tokoh dalam bentuk vertical stacked bar chart,
    diurutkan dari jumlah total sentimen terbanyak ke paling sedikit.

    Args:
        df (pd.DataFrame): DataFrame yang mengandung kolom 'tokoh' dan 'Sentimen'.

    Returns:
        count_df (pd.DataFrame): Tabel jumlah masing-masing sentimen per tokoh.
        fig (plotly.graph_objects.Figure): Objek visualisasi plotly.
    """
    # Warna kustom untuk masing-masing sentimen
    custom_colors = {'Negatif': '#EB5353', 'Netral': '#F5971D', 'Positif': '#36AE7C'}

    # Hitung jumlah sentimen per tokoh
    count_df = df.groupby(['tokoh', 'Sentimen']).size().reset_index(name='count')

    # Pivot data agar bisa divisualisasikan dalam bentuk stacked bar
    pivot_df = count_df.pivot(index='tokoh', columns='Sentimen', values='count').fillna(0).astype(int)

    # Tambahkan kolom total untuk pengurutan
    pivot_df['total'] = pivot_df.sum(axis=1)

    # Urutkan berdasarkan total
    pivot_df = pivot_df.sort_values(by='total', ascending=False).drop(columns='total')

    # Reset index setelah sort
    pivot_df = pivot_df.reset_index()

    # Ubah ke long format untuk visualisasi
    long_df = pivot_df.melt(id_vars='tokoh', var_name='Sentimen', value_name='count')

    # Visualisasi dengan plotly express (vertikal bar chart)
    fig = px.bar(
        long_df,
        x='tokoh',
        y='count',
        color='Sentimen',
        color_discrete_map=custom_colors,
        labels={'count': 'Jumlah', 'tokoh': 'Tokoh'}
    )

    # Update layout: hide legend, axis labels, tick labels kecuali xtick tokoh
    fig.update_layout(
        barmode='stack',
        showlegend=False,
        xaxis=dict(
            showticklabels=True,
            title='',
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            visible=False,
            showticklabels=False
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return count_df, fig

def visualize_wordcloud_by_sentiment_and_tokoh(df, tokoh, sentimen):
    """
    Membuat visualisasi WordCloud berdasarkan tokoh dan sentimen tertentu,
    dengan warna sesuai palet kustom untuk tiap kelas sentimen.

    Args:
        df (pd.DataFrame): DataFrame yang mengandung kolom 'tokoh', 'Sentimen', dan 'joined_swremove'.
        tokoh (str): Nama tokoh yang ingin divisualisasikan.
        sentimen (str): Kelas sentimen ('Positif', 'Negatif', atau 'Netral').

    Returns:
        fig (matplotlib.figure.Figure): Gambar WordCloud.
    """
    # Palet warna khusus
    custom_colors = {'Negatif': '#EB5353', 'Netral': '#F5971D', 'Positif': '#36AE7C'}
    color = custom_colors.get(sentimen, '#000000')  # fallback hitam jika tak ditemukan

    # Fungsi pewarna tunggal
    def single_color_func(*args, **kwargs):
        return color

    # Filter data
    filtered_df = df[(df['tokoh'] == tokoh) & (df['Sentimen'] == sentimen)]

    # Gabungkan semua teks
    text = ' '.join(filtered_df['joined_swremove'].astype(str))

    # Buat WordCloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        color_func=single_color_func
    ).generate(text)

    # Buat figure matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    return fig

def visualize_ngram_frequency(df, tokoh, sentimen='All', ngram='unigram', top_n=10):

    custom_colors = {
        'Negatif': '#EB5353',
        'Netral': '#F5971D',
        'Positif': '#36AE7C'
    }

    if ngram not in ['unigram', 'bigram', 'trigram']:
        raise ValueError("ngram harus salah satu dari: 'unigram', 'bigram', atau 'trigram'")

    n = {'unigram': 1, 'bigram': 2, 'trigram': 3}[ngram]

    filtered_df = df[df['tokoh'] == tokoh]
    sentimen_list = ['Positif', 'Netral', 'Negatif'] if sentimen == 'All' else [sentimen]
    filtered_df = filtered_df[filtered_df['Sentimen'].isin(sentimen_list)]

    all_data = []
    for label in sentimen_list:
        sub_df = filtered_df[filtered_df['Sentimen'] == label]['joined_swremove']
        cleaned = sub_df.apply(lambda x: re.sub(r"[\[\]',]", '', x) if isinstance(x, str) else x)
        text = ' '.join(cleaned.astype(str))
        tokens = text.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngram_list = [' '.join(ng) for ng in ngrams]
        counts = Counter(ngram_list)
        for ngram_text, freq in counts.items():
            all_data.append({'n-gram': ngram_text, 'Frekuensi': freq, 'Sentimen': label})

    ngram_df = pd.DataFrame(all_data)
    if ngram_df.empty:
        raise ValueError("Tidak ada data yang cocok untuk tokoh dan sentimen yang dipilih.")

    top_ngrams = ngram_df.groupby('n-gram')['Frekuensi'].sum().nlargest(top_n).index
    ngram_df = ngram_df[ngram_df['n-gram'].isin(top_ngrams)]

    x_order = ngram_df.groupby('n-gram')['Frekuensi'].sum().sort_values(ascending=False).index.tolist()

    fig = px.bar(
        ngram_df,
        x='n-gram',
        y='Frekuensi',
        color='Sentimen',
        labels={'Frekuensi': '', 'n-gram': ''},
        color_discrete_map=custom_colors,
        category_orders={'n-gram': x_order}
    )

    fig.update_layout(
        barmode='stack' if sentimen == 'All' else 'relative',
        showlegend=False,
        xaxis=dict(
            title='',
            showticklabels=True,
            tickangle=45,  # rotasi label jika panjang
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            visible=False,
            showticklabels=False,
            title=''
        ),
        margin=dict(l=0, r=0, t=0, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return ngram_df, fig

def plot_hashtag_wordcloud_by_sentiment(
    df, sentiment_filter='All', tokoh_filter=None,
    column='hashtag', width=800, height=400
):
    """
    Menampilkan wordcloud hashtag berdasarkan kolom hashtag (string) dan filter sentimen serta tokoh.
    
    Returns:
    - df_freq: DataFrame frekuensi hashtag
    - fig: Objek matplotlib Figure
    """
    filtered_df = df.copy()

    # Filter berdasarkan sentimen
    if sentiment_filter != 'All':
        filtered_df = filtered_df[filtered_df['Sentimen'] == sentiment_filter]
        color = custom_colors.get(sentiment_filter)
    else:
        color = None

    # Filter berdasarkan tokoh (opsional)
    if tokoh_filter is not None:
        filtered_df = filtered_df[filtered_df['tokoh'] == tokoh_filter]

    # Gabungkan semua hashtag
    all_hashtags = []
    for tags in filtered_df[column]:
        if isinstance(tags, str):
            all_hashtags.extend(tags.split())

    # Hitung frekuensi
    hashtag_counts = Counter(all_hashtags)
    if not hashtag_counts:
        print("Tidak ada hashtag untuk divisualisasikan.")
        return pd.DataFrame(columns=['Hashtag', 'Frekuensi']), None

    # Fungsi warna dinamis
    def color_func(word, **kwargs):
        return color if color else random.choice(list(custom_colors.values()))

    # Buat WordCloud
    wc = WordCloud(
        width=width,
        height=height,
        background_color='white',
        collocations=False,
        color_func=color_func
    ).generate_from_frequencies(hashtag_counts)

    # Buat figure dan axes seperti fungsi A
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')

    # Dataframe frekuensi hashtag
    df_freq = pd.DataFrame(hashtag_counts.items(), columns=['Hashtag', 'Frekuensi']) \
                .sort_values(by='Frekuensi', ascending=False)

    return df_freq, fig


# ----------------------------------------------------------------------------
# Streamlit UI Code 
# ----------------------------------------------------------------------------

st.markdown("<h1 style='text-align:center;'>Analisis Sentimen Pemilihan Calon Gubernur Jawa Timur 2024</h1>", unsafe_allow_html=True)


with st.container(border=True):
    col1a, col1b, col1c, col1d = st.columns(4)
    total_data, jumlah_positif, jumlah_netral, jumlah_negatif = hitung_jumlah(df_sentimen)

    with col1a:
        st.metric("Total Data", f"🗂️ {total_data:,}", border=True)
    with col1b:
        st.metric("Sentimen Positif", f"😊 {jumlah_positif:,}", border=True)
    with col1c:
        st.metric("Sentimen Netral", f"😐 {jumlah_netral:,}", border=True)
    with col1d:
        st.metric("Sentimen Negatif", f"😠 {jumlah_negatif:,}", border=True)

st.markdown("""
    <style>
    /* Membuat tab memenuhi lebar container */
    div[data-testid="stTabs"] {
        width: 100% !important;
    }
    
    /* Membuat setiap tab item memenuhi ruang */
    div[data-testid="stTabs"] button {
        flex: 1;
        min-width: 50px;
        text-align: center;
    }
    
    /* Hapus padding default */
    div[data-testid="stTabs"] > div {
        padding: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

tab1a, tab1b, tab1c = st.tabs([
    "📊 Distribusi Sentimen", 
    "📝 Analisis Teks", 
    "👥 Analisis Pengguna"
])

with tab1a:
    with st.container(border=True):
        st.subheader("Distribusi Sentimen Masing-Masing Calon Gubernur")
        df_dist_bar_counts, fig_dist_bar = visualize_sentiment_distribution_by_tokoh(df_sentimen)
        st.plotly_chart(fig_dist_bar, use_container_width=True)

with tab1b:
    with st.container(border=True):
        st.subheader(f"Word Cloud Masing-Masing Sentimen - {select_cagub}")
        col2a, col2b, col2c = st.columns(3, border=True)
        
        with col2a:
            st.markdown("<div style='text-align: center; font-size: 1rem;'>Sentimen Positif</div>", unsafe_allow_html=True)
            fig_wc_luluk_pos = visualize_wordcloud_by_sentiment_and_tokoh(df_sentimen, tokoh=select_cagub, sentimen='Positif')
            st.pyplot(fig_wc_luluk_pos, use_container_width=True)
        with col2b:
            st.markdown("<div style='text-align: center; font-size: 1rem;'>Sentimen Netral</div>", unsafe_allow_html=True)
            fig_wc_luluk_net = visualize_wordcloud_by_sentiment_and_tokoh(df_sentimen, tokoh=select_cagub, sentimen='Netral')
            st.pyplot(fig_wc_luluk_net, use_container_width=True)  
        with col2c:
            st.markdown("<div style='text-align: center; font-size: 1rem;'>Sentimen Negatif</div>", unsafe_allow_html=True)
            fig_wc_luluk_neg = visualize_wordcloud_by_sentiment_and_tokoh(df_sentimen, tokoh=select_cagub, sentimen='Negatif')
            st.pyplot(fig_wc_luluk_neg, use_container_width=True)
        
    with st.container(border=True):
        st.subheader(f"Frekuensi Penggunaan Kata Berdasarkan Sentimen - {select_cagub}")
        col3a, col3b, col3c = st.columns(3, border=True)

        with col3a:
            st.markdown("<div style='text-align: center; font-size: 1rem'>Frekuensi Unigram (1-kata)</div>", unsafe_allow_html=True)
            ngram_df_1, fig_1 = visualize_ngram_frequency(df_sentimen, tokoh=select_cagub, sentimen=label_sentimen, ngram='unigram', top_n=top_number)
            st.plotly_chart(fig_1, use_container_width=True)
        with col3b:
            st.markdown("<div style='text-align: center; font-size: 1rem'>Frekuensi Bigram (2-kata)</div>", unsafe_allow_html=True)  
            ngram_df_2, fig_2 = visualize_ngram_frequency(df_sentimen, tokoh=select_cagub, sentimen=label_sentimen, ngram='bigram', top_n=top_number)
            st.plotly_chart(fig_2, use_container_width=True)
        with col3c:
            st.markdown("<div style='text-align: center; font-size: 1rem'>Frekuensi Trigram (3-kata)</div>", unsafe_allow_html=True)
            ngram_df_3, fig_3 = visualize_ngram_frequency(df_sentimen, tokoh=select_cagub, sentimen=label_sentimen, ngram='trigram', top_n=top_number)
            st.plotly_chart(fig_3, use_container_width=True)

with tab1c:
    with st.container(border=True):
        st.subheader(f"Hashtag Terpopuler Berdasarkan Sentimen - {select_cagub}")

        with st.container(border=True):
            col5a, col5b = st.columns([3,1], vertical_alignment="center")

            with col5a:
                hashtag_df, fig_hashtag = plot_hashtag_wordcloud_by_sentiment(df_sentimen, sentiment_filter=label_sentimen, tokoh_filter=select_cagub)
                st.pyplot(fig_hashtag, use_container_width=True)
            with col5b:
                st.dataframe(hashtag_df, hide_index=True, use_container_width=True)

with st.container(border=True):
    st.subheader("Pencarian Data")

    # Input pencarian
    search_text = st.text_input("🔍 Cari Teks atau Username:", 
                                placeholder="Masukkan kata kunci atau username...")

    # Salin dan siapkan dataframe
    filtered_df = df_sentimen.copy()

    # Filter berdasarkan teks pencarian (dari kolom 'full_text' atau 'username')
    if search_text:
        filtered_df = filtered_df[
            filtered_df['full_text'].str.contains(search_text, case=False, na=False) |
            filtered_df['username'].str.contains(search_text, case=False, na=False)
        ]

        # Menentukan jenis pesan berdasarkan jumlah hasil
        result_count = len(filtered_df)
        if result_count == 0:
            st.info("Tidak ada data yang cocok ditemukan.")
        else:
            st.success(f"Ditemukan {result_count} data yang cocok.")

    # Pilih dan ubah nama kolom yang ingin ditampilkan
    display_df = filtered_df[['username', 'full_text', 'Sentimen']].rename(
        columns={
            'username': 'Username',
            'full_text': 'Full Text'
        }
    )

    # Tampilkan dalam Streamlit
    st.dataframe(display_df, hide_index=True, use_container_width=True,
                 column_config={"Username": st.column_config.Column(width="small"),
                                "Full Text": st.column_config.Column(width="large"),
                                "Sentimen": st.column_config.Column(width="small")})