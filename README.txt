# Sentimen Pemilihan Calon Gubernur Jawa Timur 2024 Dashboard

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun sebuah dashboard interaktif yang menampilkan analisis dan prediksi sentimen terkait pemilihan Calon Gubernur Jawa Timur 2024. Dashboard ini menyajikan visualisasi data sentimen dari media sosial terhadap tiga calon utama: Luluk Nur Hamidah, Khofifah Indar Parawansa, dan Tri Rismaharini. 

Selain menampilkan visualisasi seperti distribusi sentimen, WordCloud, dan analisis N-gram, pengguna juga dapat melakukan prediksi sentimen dari teks yang dimasukkan secara manual. Model prediksi yang digunakan adalah algoritma klasifikasi Random Forest, yang telah dilatih menggunakan data sentimen dari media sosial yang telah diproses dan dilabeli.

Dashboard ini dibangun menggunakan bahasa pemrograman Python dan memanfaatkan berbagai library seperti Streamlit untuk antarmuka interaktif, scikit-learn untuk pemodelan machine learning, serta pandas, NLTK, Plotly, dan WordCloud untuk pemrosesan dan visualisasi data.

## Struktur Folder
📦Sentimen Cagub
┣ 📂Dashboard
┃ ┣ 📂app-pages
┃ ┃ ┣ 📜page_dashboard.py — halaman Streamlit untuk visualisasi analisis sentimen.
┃ ┃ ┗ 📜page_prediksi.py — halaman Streamlit untuk prediksi sentimen dari input teks pengguna.
┃ ┗ 📜sentimen_cagub_app.py — file utama untuk menjalankan aplikasi Streamlit.
┣ 📂Data
┃ ┣ 📜data_cagub_analisis.csv — dataset utama hasil penggabungan dan pembersihan data dari ketiga calon gubernur.
┃ ┣ 📜Kamus Normalisasi.csv — kamus kata alay untuk proses normalisasi teks.
┃ ┣ 📜ReLabeling - Gabungan.csv — data gabungan dari semua cagub.
┃ ┣ 📜ReLabeling - Khofifah.csv — data sentimen khusus Khofifah.
┃ ┣ 📜ReLabeling - Luluk.csv — data sentimen khusus Luluk.
┃ ┗ 📜ReLabeling - Risma.csv — data sentimen khusus Risma.
┣ 📂Model
┃ ┣ 📜best_saved_rf_model.pkl — model klasifikasi Random Forest yang telah dilatih, dengan akurasi tertinggi.
┃ ┣ 📜best_saved_selector.pkl — objek selektor fitur yang disimpan setelah proses seleksi fitur menggunakan Mutual Information.
┃ ┗ 📜best_saved_tfidf_vectorizer.pkl — vectorizer TF-IDF yang digunakan untuk mengubah teks menjadi fitur numerik saat pelatihan model.
┣ 📂Python Notebook
┃ ┣ 📜[Update]_Sentimen_Cagub_Jatim_2024_original.ipynb — notebook menggunakan data asli untuk training model.
┃ ┣ 📜[Update]_Sentimen_Cagub_Jatim_2024_sampling.ipynb — notebook dengan proses sampling data untuk training model.
┃ ┗ 📜[Update]_Sentimen_Cagub_Jatim_2024_visualisasi.ipynb — notebook berisi eksplorasi dan visualisasi data.
┣ 📜README.txt — informasi umum dan petunjuk penggunaan proyek.
┗ 📜requirement.txt — daftar dependensi dan versi library yang digunakan dalam proyek.

## Instalasi dan Pengaturan
1. **Instalasi Dependensi:**  
Pastikan Anda telah menginstal semua dependensi yang dibutuhkan (lihat file requirement.txt). Daftar dependensi utama yang digunakan dalam proyek ini beserta versinya:
- pandas==2.2.3 (untuk manipulasi data)
- scikit-learn==1.6.1 (untuk pemodelan dan evaluasi machine learning)
- Sastrawi==1.0.1 (untuk stemming bahasa Indonesia)
- matplotlib==3.5.2 (untuk visualisasi dasar)
- plotly==5.22.0 (untuk visualisasi interaktif)
- wordcloud==1.9.4 (untuk visualisasi frekuensi kata)
- streamlit==1.41.1 (untuk membangun antarmuka aplikasi web)
- nltk==3.9.1 (untuk stopword dan preprocessing teks)

2. **Menjalankan Aplikasi:**  
> Untuk menjalankan aplikasi:
- Buka terminal/command prompt
- Navigasi ke direktori utama proyek
- Jalankan perintah: ``streamlit run Dashboard/sentimen_cagub_app.py`` Aplikasi akan otomatis terbuka di browser default (biasanya di http://localhost:8501)

> Untuk menghentikan server Streamlit:
- Kembali ke terminal tempat aplikasi dijalankan
- Tekan kombinasi tombol: Windows/Linux: ``Ctrl + C`` | MacOS: ``Command + C``
- Tunggu hingga proses benar-benar berhenti (terminal akan menampilkan pesan "Server stopped")

3. **Memuat Model dan Data:**  
Aplikasi ini sudah terintegrasi dengan model Random Forest yang disimpan dalam format `.pkl` bersama dengan TF-IDF vectorizer untuk pemrosesan teks. Pastikan Anda menyimpan file `saved_rf_model_8291_acc.pkl` dan `saved_tfidf_vectorizer_80_new.pkl` di dalam folder `Model`.

4. **Penggunaan:**  
- **Halaman Dashboard:** Menampilkan informasi visualisasi dan statistik terkait Pemilihan Calon Gubernur Jawa Timur 2024 (Disertai filter yang dapat digunakan).
- **Halaman Prediksi Sentimen:** Pengguna dapat memasukkan teks dan mengklik tombol "Prediksi Sentimen". Aplikasi akan memberikan hasil prediksi sentimen dan confidence score untuk setiap kelas (Positif, Netral, Negatif).
