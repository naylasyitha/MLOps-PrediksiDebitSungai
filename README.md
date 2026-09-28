# MLOps-PrediksiDebitSungai

## Deskripsi

Repositori ini merupakan fondasi teknis untuk proyek MLOps yang bertujuan membangun sistem prediksi debit sungai sebagai upaya deteksi dini risiko banjir di Jakarta. Proyek ini menerapkan pendekatan **Continual Learning / Continuous Training (CL/CT)** dengan memanfaatkan data time series dari **Open-Meteo Flood API**.

Environment pengembangan dikonfigurasi menggunakan **GitHub Codespaces** agar proyek bersifat reproducible dan dapat dijalankan ulang secara konsisten di lingkungan mana pun, dengan manajemen kode dan eksperimen mengikuti strategi **GitHub Flow**.

## Tujuan Proyek

- Membangun sistem prediksi debit sungai untuk deteksi dini risiko banjir di Jakarta.
- Menerapkan environment pengembangan yang terstandar dan reproducible menggunakan GitHub Codespaces.
- Mengelola kode dan eksperimen menggunakan strategi GitHub Flow.
- Membangun fondasi awal menuju penerapan pipeline Continual Learning / Continuous Training.

## Struktur Direktori

```
MLOps-PrediksiDebitSungai/
├── .devcontainer/
│   └── devcontainer.json
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── ingest_data.py
│   └── preprocess.py
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

- `pandas`, `numpy` — manipulasi dan analisis data
- `scikit-learn` — pemodelan machine learning
- `matplotlib` — visualisasi data
- `requests` — pengambilan data dari Open-Meteo Flood API
- `jupyter` — eksplorasi data interaktif

Python yang digunakan: **3.11**.

## Pipeline Data

Pipeline ini mengambil data hidrometeorologis Kota Jakarta dari Open-Meteo secara berkala, menyimpannya sebagai data mentah bertimestamp, lalu membersihkannya menjadi dataset siap pakai untuk Continual Learning.

**Alur:**

```
Open-Meteo API → src/ingest_data.py → data/raw/ → src/preprocess.py → data/processed/
```

### Instalasi Dependensi

```bash
pip install -r requirements.txt
```

---

## 1. Menjalankan Ingestion (`src/ingest_data.py`)

```bash
python src/ingest_data.py
```

**Argumen opsional:**

- `--start-date YYYY-MM-DD` — tanggal awal backfill. Hanya berlaku pada run pertama (default: `2024-01-01`).
- `--output-dir PATH` — folder tujuan file raw (default: `data/raw`).

**Sumber data yang diambil:**

| Sumber   | Endpoint Open-Meteo              | Blok data | Keterangan |
|----------|----------------------------------|-----------|------------|
| flood    | `flood-api.open-meteo.com`       | daily     | `river_discharge` |
| weather  | `archive-api.open-meteo.com`     | hourly    | `precipitation`, `soil_moisture_0_to_7cm`, timezone `Asia/Jakarta` |
| forecast | `api.open-meteo.com`             | hourly    | `precipitation`, `forecast_days=7`, timezone `Asia/Jakarta` |

**Perilaku:**

- **Run pertama** mengambil data historis sejak `--start-date` (**backfill**).
- **Run berikutnya** hanya mengambil data sejak tanggal terakhir **per sumber** yang sudah ada, ditambah **overlap 7 hari** untuk menangkap revisi nilai.
- **Batas akhir pengambilan** adalah **H-6** (`ARCHIVE_DELAY_DAYS = 6`) untuk menghindari data arsip yang belum final.
- Setiap run membuat **file baru bertimestamp UTC**; file lama tidak pernah ditimpa.
- Kegagalan koneksi diulang hingga **3 kali** dengan exponential backoff. Retry hanya berlaku untuk status `429, 500, 502, 503, 504`.
- Jika sebuah sumber tetap gagal, sumber lain tetap disimpan dan skrip berakhir dengan **exit code 1**.

> **Catatan:** Data `forecast` disimpan sebagai snapshot untuk keperluan inference/forecasting di tahap berikutnya. Saat ini `forecast` **belum dipakai** di pipeline prapemrosesan.

### Format Data Mentah

Data mentah tersimpan di `data/raw/` dengan pola nama:

```
data_YYYYMMDD_HHMMSS_<sumber>.csv
```

| Sumber   | Kolom |
|----------|-------|
| flood    | `time` (tanggal), `river_discharge` (m³/s), `ingested_at` |
| weather  | `time` (per jam, WIB), `precipitation` (mm), `soil_moisture_0_to_7cm` (m³/m³), `ingested_at` |
| forecast | `time` (per jam, WIB), `precipitation` (mm), `ingested_at` |

---

## 2. Menjalankan Prapemrosesan (`src/preprocess.py`)

```bash
python src/preprocess.py
```

**Argumen opsional:**

- `--raw-dir PATH` — folder sumber file raw (default: `data/raw`)
- `--processed-dir PATH` — folder tujuan dataset (default: `data/processed`)

**Tahapan:**

1. **Muat & gabungkan** semua file raw **per sumber** (`flood` dan `weather`). Data `forecast` belum dipakai di sini.
2. **Buang duplikat `time`** — data dengan `ingested_at` terbaru dipertahankan.
3. **Validasi nilai fisik** — nilai di luar batas diubah menjadi `NaN`:
   - `river_discharge` ≥ 0
   - `precipitation` ≥ 0
   - `soil_moisture` ∈ [0, 1]
4. **Agregasi harian weather** — hujan dijumlah, soil moisture dirata-rata. Hari yang tidak lengkap 24 jam dibuang.
5. **Gabungkan** data debit dan cuaca berdasarkan tanggal (`inner join`).
6. **Lengkapi indeks harian** (`reindex` ke rentang tanggal penuh) lalu isi nilai kosong:
   - Interpolasi waktu untuk celah **≤ 3 hari** pada `river_discharge` dan `soil_moisture`.
   - Celah > 3 hari dibiarkan kosong.
   - `precipitation` kosong diisi `0`.
7. **Rekayasa fitur**:
   - Lag debit: `river_discharge_lag1`, `lag2`, `lag3`
   - Rolling hujan: `precipitation_roll3`, `precipitation_roll7`
   - Rolling soil moisture: `soil_moisture_roll3`
   - Fitur musiman: `month`, `is_rainy_season` (1 jika bulan ∈ {11, 12, 1, 2, 3})
8. **Buang baris** yang masih memiliki nilai kosong (termasuk celah > 3 hari yang sengaja tidak diinterpolasi) dengan `dropna()`.
9. **Peringatan** jika jumlah baris < 500 (`MIN_ROWS_FOR_DVC`), disarankan untuk LK-05.
10. **Simpan** dataset ke `data/processed/dataset_YYYYMMDD_HHMMSS.csv` tanpa menimpa file lama.

> Prapemrosesan disesuaikan dengan domain data deret waktu hidrologi, bukan teks, sehingga tahapnya berupa pembersihan nilai, agregasi temporal, dan rekayasa fitur waktu.

### Format Dataset Akhir

File: `data/processed/dataset_YYYYMMDD_HHMMSS.csv`

Kolom:

```
date, river_discharge, precipitation, soil_moisture,
river_discharge_lag1, river_discharge_lag2, river_discharge_lag3,
precipitation_roll3, precipitation_roll7, soil_moisture_roll3,
month, is_rainy_season
```

---

## 3. Penjadwalan Periodik

Pengambilan berkala dapat dijadwalkan dengan **cron** di server Linux, misalnya setiap hari pukul 07.00:

```bash
0 7 * * * cd /path/ke/MLOps-PrediksiDebitSungai && python src/ingest_data.py && python src/preprocess.py
```

> **Catatan:** Karena menggunakan `&&`, jika `ingest_data.py` mengembalikan **exit code 1** (ada sumber yang gagal), maka `preprocess.py` **tidak akan dijalankan**. Ini disengaja agar dataset tidak dibentuk dari data yang tidak lengkap. Jika ingin tetap menjalankan prapemrosesan meski ingestion parsial gagal, ganti `&&` menjadi `;`.

Di **GitHub Codespaces**, simulasi periodik dilakukan dengan menjalankan ulang kedua skrip secara manual:

```bash
python src/ingest_data.py && python src/preprocess.py
```

---

## 4. Standar Kode

```bash
pip install black flake8
black --line-length 79 src/
flake8 --max-line-length 79 src/
```