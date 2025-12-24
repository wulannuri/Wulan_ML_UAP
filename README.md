# Deteksi Ekpresi Wajah (disgust, sad dan happy)

# Deskripsi Projek
Proyek ini merupakan aplikasi **deteksi emosi berdasarkan ekspresi wajah** yang menggunakan metode **Deep Learning** untuk mengklasifikasikan emosi manusia dari citra wajah. Sistem dibangun dengan memanfaatkan **Convolutional Neural Network (CNN)** menggunakan beberapa arsitektur model, yaitu **Base CNN, VGG19, dan MobileNet**. Aplikasi berbasis web ini dikembangkan menggunakan **Streamlit** sehingga pengguna dapat mengunggah gambar wajah dan memperoleh hasil prediksi emosi berupa label **disgust, happy, atau sad** beserta nilai probabilitasnya. Proyek ini bertujuan untuk menerapkan konsep **computer vision** dalam analisis emosi manusia secara otomatis dan interaktif.    


# Latar Belakang
Ekspresi wajah merupakan salah satu bentuk komunikasi nonverbal yang paling penting dalam menyampaikan emosi manusia. Kemampuan untuk mengenali emosi melalui ekspresi wajah memiliki peran yang signifikan dalam berbagai bidang, seperti interaksi manusia dan komputer, psikologi, pendidikan, serta sistem keamanan. Namun, proses identifikasi emosi secara manual masih bergantung pada pengamatan manusia yang bersifat subjektif dan tidak selalu konsisten.
Perkembangan teknologi Artificial Intelligence, khususnya pada bidang Deep Learning dan Computer Vision, memungkinkan pengolahan citra wajah secara otomatis untuk mengenali pola-pola ekspresi emosi dengan tingkat akurasi yang tinggi. Convolutional Neural Network (CNN) menjadi salah satu metode yang paling efektif dalam melakukan ekstraksi fitur dari citra wajah dan mengklasifikasikannya ke dalam kategori emosi tertentu.
Berdasarkan hal tersebut, proyek ini dikembangkan untuk membangun sebuah sistem deteksi emosi berdasarkan ekspresi wajah dengan memanfaatkan beberapa arsitektur CNN, yaitu Base CNN, VGG19, dan MobileNet. Sistem ini diimplementasikan dalam bentuk aplikasi web berbasis Streamlit agar mudah digunakan dan dapat diakses secara interaktif oleh pengguna. Diharapkan aplikasi ini dapat menjadi media pembelajaran serta contoh penerapan deep learning dalam pengenalan emosi manusia.


# Tujuan Pengembangan
1. Mengembangkan sistem deteksi emosi berdasarkan ekspresi wajah menggunakan metode Deep Learning.
2. Menerapkan serta membandingkan performa beberapa arsitektur Convolutional Neural Network (CNN), yaitu Base CNN, VGG19, dan MobileNet.
3. Mengklasifikasikan emosi wajah ke dalam tiga kategori, yaitu disgust, happy, dan sad.
4. Mengimplementasikan sistem dalam bentuk aplikasi web berbasis Streamlit yang interaktif dan mudah digunakan.
5. Menjadi sarana pembelajaran dalam penerapan konsep computer vision dan machine learning pada pengolahan citra wajah.


# Preprocessing
Preprocessing data citra diimplementasikan secara modular pada file data/preprocessing.py. Proses meliputi resizing citra menjadi 128×128 piksel, normalisasi nilai piksel, serta augmentasi data. Untuk model VGG19 dan MobileNetV2 digunakan fungsi preprocess_input bawaan TensorFlow sesuai karakteristik model pretrained.

# 📊 Sumber Dataset 📊
Dataset yang digunakan dalam projek ini berasal dari platfrom yang 
bersifat open source bernama kaggle. Peneliti menggunakan dataset berjudul MMA 
Facial Expression Dataset (https://www.kaggle.com/datasets/mahmoudima/mma
facial-expression). Dari keseluruhan data yang tersedia, peneliti menggunakan data 
sebanyak 3.015 citra wajah manusia. Data tersebut diklasifikasikan ke dalam tiga 
kelas, yaitu happy, sad, dan disgust. 

# PEMODELAN
Model yang digunakan dalam proyek ini meliputi:
1. Base Convolutional Neural Network (Base CNN)
Base CNN merupakan model jaringan saraf konvolusional sederhana yang dibangun dari beberapa lapisan Convolution, Pooling, dan Fully Connected. Model ini digunakan sebagai model dasar untuk memahami proses ekstraksi fitur dari citra wajah. Base CNN memiliki arsitektur yang relatif ringan sehingga mudah dilatih dan cepat dalam proses inferensi. Model ini berfungsi sebagai pembanding awal terhadap model yang memiliki arsitektur lebih kompleks.

3. VGG19
VGG19 merupakan salah satu arsitektur Deep Convolutional Neural Network yang memiliki 19 lapisan, terdiri dari lapisan konvolusi dan fully connected. Model ini dikenal mampu mengekstraksi fitur citra secara mendalam dan detail. Dalam proyek ini, VGG19 digunakan untuk meningkatkan kemampuan model dalam mengenali pola ekspresi wajah yang lebih kompleks. Model ini memerlukan proses preprocessing khusus serta sumber daya komputasi yang lebih besar dibandingkan Base CNN.

4. MobileNet
MobileNet adalah arsitektur CNN yang dirancang agar ringan dan efisien dengan menggunakan teknik depthwise separable convolution. Model ini cocok digunakan pada aplikasi dengan keterbatasan sumber daya komputasi. Dalam proyek ini, MobileNet digunakan untuk menghasilkan sistem deteksi emosi yang tetap memiliki performa baik namun dengan waktu inferensi yang lebih cepat dan penggunaan memori yang lebih rendah dibandingkan VGG19.



# Hasil dan Analisis Perbandingan
1. Base CNN
   
<img width="1200" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/3cda715d-b8de-4443-8a9b-5fafa7401827" />

Berdasarkan grafik hasil pelatihan model Base CNN, akurasi data training meningkat secara konsisten, menunjukkan bahwa model mampu mempelajari pola pada data latih dengan baik. Namun, akurasi pada data validasi cenderung lebih rendah dan tidak meningkat secara signifikan, yang mengindikasikan keterbatasan model dalam melakukan generalisasi.
Selain itu, training loss mengalami penurunan yang jelas, sedangkan validation loss relatif stagnan. Kondisi ini menunjukkan adanya overfitting, di mana model terlalu menyesuaikan diri dengan data training. Oleh karena itu, Base CNN cukup baik sebagai model baseline, tetapi diperlukan metode yang lebih kuat seperti transfer learning untuk meningkatkan performa pada data validasi.

📊 Classification Report

|  | Precision | recall | F1-score |
|--------|---------|--------|---------|
| disgust | 0.00 | 0.00 | 0.00  |
| happy |  0.67 | 0.92 | 0.78  |
| sad | 0.40  | 0.15 | 0.22  |
| accuracy | | | 0.65  |


📊 Confusion Matrix 

<img width="600" height="500" alt="Figure_4" src="https://github.com/user-attachments/assets/42006051-87de-48e7-a44c-7b5e340bcaeb" />




2. VGG16

<img width="1200" height="400" alt="Figure_2" src="https://github.com/user-attachments/assets/7a27cf1f-ba40-44af-8899-0f3e60fa5acb" />

Berdasarkan grafik VGG19, akurasi data training meningkat secara konsisten, sedangkan akurasi validasi justru menurun pada epoch akhir. Hal ini menunjukkan bahwa model mampu mempelajari data latih dengan baik, tetapi kurang mampu melakukan generalisasi pada data validasi. Pola ini diperkuat oleh grafik loss, di mana loss training terus menurun, sementara loss validasi kembali meningkat, yang merupakan indikasi overfitting. Dengan demikian, meskipun VGG19 efektif dalam menangkap pola pada data latih, performanya pada data baru masih kurang optimal untuk klasifikasi emosi disgust, happy, dan sad.

📊 Classification Report

|  | Precision | recall | F1-score |
|--------|---------|--------|---------|
| disgust | 0.00 | 0.00 | 0.00  |
| happy |  0.66 | 1.00 | 0.79  |
| sad | 0.00  | 0.00 | 0.00  |
| accuracy | | | 0.66  |


📊 Confusion Matrix 

<img width="600" height="500" alt="Figure_5" src="https://github.com/user-attachments/assets/f79e0dd8-4b30-4729-b842-2d0683f5eca4" />




3. MobileNet

<img width="1200" height="400" alt="Figure_3" src="https://github.com/user-attachments/assets/00935f25-86b8-4245-bf7f-a5faa7e42494" />

Grafik VGG19 menunjukkan bahwa akurasi training meningkat stabil, sedangkan akurasi validasi cenderung menurun, terutama pada epoch terakhir. Hal ini mengindikasikan bahwa model belajar dengan baik pada data latih tetapi kurang mampu melakukan generalisasi pada data validasi. Pola tersebut diperkuat oleh grafik loss, di mana loss training terus menurun, sementara loss validasi kembali meningkat, yang merupakan tanda terjadinya overfitting pada model VGG19.

📊 Classification Report

|  | Precision | recall | F1-score |
|--------|---------|--------|---------|
| disgust | 0.00 | 0.00 | 0.00  |
| happy |  0.75 | 0.75 | 0.75  |
| sad | 0.40  | 0.52 | 0.45  |
| accuracy | | | 0.63  


📊 Confusion Matrix 

<img width="600" height="500" alt="Figure_6" src="https://github.com/user-attachments/assets/2a6b2388-144f-4738-89c6-0d76ab75e11f" />




4. Tabel Perbandingan

| Model | Akurasi |
|--------|---------|
| Base CNN | 65%  |
| VGG19 | 66%  |
| MobileNet | 63%  |

# Panduan menjalankan sistem website secara lokal 
1. Pastikan struktur folder sudah benar
<img width="252" height="294" alt="image" src="https://github.com/user-attachments/assets/284a465e-6afa-40a7-99fa-bae84b9f189d" />

2. Buat Requirement.txt
<img width="174" height="160" alt="image" src="https://github.com/user-attachments/assets/98e422a3-8589-44d7-84af-3320cda4d6b6" />

3. kemudian pastikan semua path model sudah sesuai
4. Push folder kedalam github
<img width="449" height="111" alt="image" src="https://github.com/user-attachments/assets/76cd17f2-7f67-4d16-ad3b-9bc9cf010770" />

5. DEPLOY KE STREAMLIT CLOUD 🚀

Langkahnya:

a. Buka 👉 https://streamlit.io/cloud

b. Klik Sign in with GitHub

c. Klik New app

  Isi:
  Repository → repo Anda
  Branch → main
  Main file path → app.py
  Klik Deploy
  ⏳ Tunggu ±1–3 menit (pertama kali lebih lama)


# Link Live Demo

 https://dwulanmluap-ctwzunzjtnapyywxstwtqf.streamlit.app/

