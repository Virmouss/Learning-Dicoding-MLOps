# Submission: Sentiment Classification - Game Reviews / Steam Reviews

## [Certificate](https://www.dicoding.com/certificates/GRX5W0QMVZ0M)

| | Deskripsi |
| ----------- | ----------- |
| Dataset | Dataset yang digunakan dalam proyek ini bersumber dari [Steam Reviews Dataset](https://www.kaggle.com/datasets/luthfim/steam-reviews-dataset/data). Data ini merepresentasikan interaksi pengguna dalam bentuk ulasan teks terhadap berbagai judul game di platform Steam. Proyek ini menggunakan sampel sebanyak 100.000 baris data. Terdapat 2 fitur yang digunakan yaitu "review" sebagai fitur teks dan "sentiment" sebagai fitur kategorikal (label biner).|
| Masalah | Dalam industri video game, volume ulasan Steam yang masif dan tidak terstruktur membuat analisis manual menjadi tidak efektif, sehingga keluhan teknis yang kritis sering kali tertimbun oleh noise (seperti spam atau ASCII art) dan berisiko merugikan reputasi bisnis. Untuk mengatasi hal ini, proyek ini mengimplementasikan model klasifikasi sentimen sebagai sistem peringatan dini yang mampu mengekstrak opini mentah menjadi metrik kepuasan pengguna secara otomatis dan real-time. Tingginya tingkat noise pada data ulasan tersebut, solusi ini dibangun di atas sebuah pipeline MLOps komprehensif untuk memastikan bahwa seluruh proses mulai dari pembersihan data, penyeimbangan kelas, hingga saat deployment model dapat berjalan secara konsisten dan scalable di lingkungan produksi. |
| Solusi machine learning | Proyek ini menggunakan arsitektur Neural Network berbasis Global Average Pooling 1D. Keunggulan dengan pendekatan ini dikarenakan penggunaan daya komptasi yang efisien dibandingkan dengan model berbasis transformer seperti BERT yang cenderung berat. Model ini memiliki ukuran file yang kecil dan latensi rendah, sehingga ideal untuk diimplementasikan kepada layanan backend. Meskipun ringan, model ini efektif dalam menangani hubungan non-linear antar kata melalui lapisan Embedding yang dipelajari selama proses pelatihan, sehingga cukup kuat untuk klasifikasi sentimen biner. |
| Metode pengolahan | Dataset memasuki proses exploratory data analysis (EDA) terlebih dahulu dimana proses tersebut terdapat class balancing, language fitering, noise reduction. Lalu setelah proses EDA selesai dilakukan proses standarisasi text menggunakan Tensorflow Transform. Pada akhirnya terdapat dua fitur yang digunakan yaitu "review" (string) dan "sentiment" (int64) |
| Arsitektur model | Model menggunakan Global Average Pooling 1D neural network yang di optimasi untuk training cepat dan latensi yang pendek untuk deployment. |
| Metrik evaluasi | Metrik yang digunakan adalah Binary Crossentropy, Precision, Recall, dan F1-Score |
| Performa model | Hasil training secara keseluruhan mencapai nilai-nilai yang cukup memuaskan, model mencapai binarry accuracy sebesar 83%, Precision sebesar 84%, Recall sebesar 81%, dan F1-Score sebesar 82%. Hal tersebut menggambarkan bahwa model dapat melakukan klasifikasi review-review secara akurat dengan nilai recall dan precision yang cukup besar. Model jika diberikan teks review yang bersifat ambigu saat melakukan inference terhadap model akan menghasilkan nilai yang mengambang di kisaran 0.5, hal tersebut juga dapat menjadi kelemahan model karena fitur "sentiment" yang hanya bersifat biner yaitu Recommended atau Not Recommended |
| Opsi Deployment | Model di deploy menggunakan TF Serving melalui platform Railway |
| Web App | [review-classification](https://izzan-virm-review-classification-production.up.railway.app/v1/models/review-classification-model/metadata), [Inference Endpoint](izzan-virm-review-classification-production.up.railway.app/v1/models/review-classification-model:predict) 
| Monitoring | Prometheus dapat memonitor jumlah request yang telah dijalankan oleh user terhadap endpoint prediction.|

- Arsitektur Model

| Layer | Neuron | Activation |
| ----------- | ----------- | ----------- |
| Input Layer | (None, 1) | -
| Vectorization | (None, 100) | -
| Embedding | (None, 100, 32) | -
| Global Average Pooling 1D | (None, 48)| -
| Dense | (None, 32) | ReLU
| Dense | (None, 32) | ReLU
| Dense | (None, 16) | ReLU
| Output Layer | (None, 1) | Sigmoid

| Hyperparameter | Detail |
| ----------- | ----------- |
| Optimizer | Adam (Learning Rate: 0.0001)
| Loss Functions | Binary Crossentropy
| Batch Size | 64
| Epochs | 5

- Performa Model

| Metrik | Nilai |
| ----------- | ----------- |
| Binary Accuracy | 83%
| Loss | 0.463
| Precision | 84%
| Recall | 81%
| F1-Score | 82%