# MLOps-PrediksiDebitSungai

## Deskripsi
Repositori ini merupakan fondasi teknis untuk proyek MLOps yang bertujuan membangun sistem prediksi debit sungai sebagai upaya deteksi dini risiko banjir di Jakarta. Proyek ini menerapkan pendekatan Continual Learning/Continuous Training (CL/CT) dengan memanfaatkan data time series dari Open-Meteo Flood API.

Environment pengembangan dikonfigurasi menggunakan GitHub Codespaces agar proyek bersifat reproducible dan dapat dijalankan ulang secara konsisten di lingkungan mana pun, dengan manajemen kode dan eksperimen mengikuti strategi GitHub Flow.

## Tujuan Proyek
- Membangun sistem prediksi debit sungai untuk deteksi dini risiko banjir di Jakarta
- Menerapkan environment pengembangan yang terstandar dan reproducible menggunakan GitHub Codespaces
- Mengelola kode dan eksperimen menggunakan strategi GitHub Flow
- Membangun fondasi awal menuju penerapan pipeline Continual Learning/Continuous Training

## Struktur Direktori
```
MLOps-PrediksiDebitSungai/
├── .devcontainer/
│ └── devcontainer.json
├── data/
│ ├── raw/
│ └── processed/
├── models/ 
├── notebooks/
├── src/
├── config/
├── tests/
├── docs/
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Dependencies
Library utama yang digunakan tercantum di `requirements.txt`, meliputi:
- pandas, numpy - manipulasi dan analisis data
- scikit-learn - pemodelan machine learning
- matplotlib - visualisasi data
- requests - pengambilan data dari Open-Meteo Flood API
- jupyter - eksplorasi data interaktif
