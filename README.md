# Identitas Pembuat
* Nama: Anisa Julianti
* NPM: G1A023052
* Program Studi: Informatika, Universitas Bengkulu

---

# Studi Kasus Skripsi: Klasifikasi Otomatis Penyebab Deforestasi Menggunakan Convolutional Neural Network (CNN) pada Citra Satelit

Repositori ini berisi implementasi kode program Python (`implementasi_materi6.py`) untuk membangun dan melatih model *Deep Learning* berbasis *Convolutional Neural Network* (CNN). Sistem ini dirancang untuk mengklasifikasikan pemicu atau penyebab langsung (*direct drivers*) deforestasi berdasarkan pola spasial dan spektral citra satelit (simulasi data multispektral Sentinel).

---

## 1. Latar Belakang dan Permasalahan
* Isu Lingkungan: Peringatan dini kehilangan hutan (*deforestation alerts*) umumnya hanya menunjukkan titik lokasi kerusakan hutan (*di mana*), tanpa memberikan informasi mengenai faktor pemicunya (*mengapa*).
* Solusi Berbasis IT: Mengembangkan model *Computer Vision* menggunakan arsitektur CNN untuk mengenali pola visual (seperti tekstur, bentuk bukaan lahan, dan respons spektral) guna mengotomatisasi atribusi penyebab deforestasi secara cepat.
* Tujuan Proyek (`implementasi_materi6a.py`):
  1. Membangkitkan dataset simulasi citra satelit beresolusi 128x128 piksel dengan pola spasial unik untuk 4 kategori kelas (Perkebunan Ilegal, Pertanian Rakyat, Kebakaran Hutan, dan Tambang Terbuka).
  2. Membangun arsitektur *Deep Convolutional Neural Network* bertingkat lengkap dengan teknik regularisasi (*BatchNormalization* dan *Dropout*).
  3. Menerapkan strategi pelatihan tingkat lanjut menggunakan *callbacks* (`EarlyStopping` dan `ReduceLROnPlateau`).
  4. Mengevaluasi performa model menggunakan metrik klasifikasi komprehensif (*Accuracy, Precision, Recall, F1-Score*).

---

## 2. Alur Kerja Sistem (Pipeline)
Skrip ini mengikuti tahapan standar riset *Machine Learning* dan *Deep Learning*:
1. **Pembangkitan Dataset Sintetis:** Mensimulasikan matriks citra 3-kanal (RGB/Spektral) dengan karakteristik tekstur spesifik untuk setiap kelas pemicu deforestasi.
2. **Pembagian Dataset (*Data Splitting*):** Membagi data secara proporsional dan berstrata (*stratified split*) menjadi Data Latih (Training), Data Validasi (Validation), dan Data Uji (Testing).
3. **Konstruksi Model CNN:** Menyusun lapisan konvolusi berjenjang (`Conv2D`), *pooling*, dan *fully-connected layers* untuk ekstraksi fitur spasial otomatis.
4. **Pelatihan (*Training*) & Optimasi:** Melatih model menggunakan *optimizer* Adam dengan pemantauan metrik validasi secara real-time.
5. **Evaluasi Model:** Menguji kemampuan generalisasi model terhadap data uji yang belum pernah dilihat sebelumnya.

---

## 3. Implementasi Kode (`implementasi_materi6.py`)

Skrip Python ini dibangun menggunakan pustaka TensorFlow/Keras dan Scikit-Learn:

```python
# 1. Pembangkitan Dataset Sintetis Berbasis Spatial Texture
# 2. Konstruksi Arsitektur CNN Tingkat Lanjut (Deep Convolutional Network)
# 3. Pelatihan Model dengan EarlyStopping & ReduceLROnPlateau
# 4. Evaluasi Performa Model dan Laporan Klasifikasi Komprehensif
