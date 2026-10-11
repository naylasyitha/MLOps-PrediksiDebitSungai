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
├── .dvc/
│   └── config                
├── data/
│   ├── raw/                  
│   ├── processed/            
│   ├── raw.dvc               
│   ├── processed.dvc         
│   └── .gitignore            
├── models/
├── notebooks/
├── src/
│   ├── ingest_data.py
│   └── preprocess.py
├── config/
├── tests/
├── docs/
├── .dvcignore
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
- `dvc[s3]`: versioning data dengan remote S3-compatible (MinIO)

Python yang digunakan: **3.11**.

## Pipeline Data

Pipeline ini mengambil data hidrometeorologis Kota Jakarta dari Open-Meteo secara berkala, menyimpannya sebagai data mentah bertimestamp, lalu membersihkannya menjadi dataset siap pakai untuk Continual Learning.

## Alur Pipeline Data

```
Open-Meteo API → src/ingest_data.py → data/raw/ → src/preprocess.py → data/processed/ → DVC → MinIO
```

---

## 1. Ingestion (`src/ingest_data.py`)

```bash
python src/ingest_data.py
```

**Argumen opsional:**

- `--start-date YYYY-MM-DD`: tanggal awal backfill, hanya berlaku pada run pertama (default `2024-01-01`).
- `--output-dir PATH`: folder tujuan file raw (default `data/raw`).

**Sumber data:**

| Sumber   | Endpoint                     | Resolusi | Variabel |
|----------|------------------------------|----------|----------|
| flood    | `flood-api.open-meteo.com`   | harian   | `river_discharge` |
| weather  | `archive-api.open-meteo.com` | per jam  | `precipitation`, `soil_moisture_0_to_7cm` |
| forecast | `api.open-meteo.com`         | per jam  | `precipitation` (prakiraan 7 hari) |

**Perilaku:**

- **Run pertama** mengambil data historis sejak `--start-date` (backfill).
- **Run berikutnya** hanya mengambil data sejak tanggal terakhir per sumber, ditambah **overlap 7 hari** untuk menangkap revisi nilai.
- Batas akhir pengambilan adalah **hari ini dikurangi 6 hari**, karena data arsip memiliki keterlambatan.
- Setiap run membuat **file baru bertimestamp UTC**. File lama tidak pernah ditimpa.
- Kegagalan koneksi diulang hingga **3 kali** dengan exponential backoff, khusus status `429, 500, 502, 503, 504`.
- Jika satu sumber tetap gagal, sumber lain tetap disimpan dan skrip berakhir dengan **exit code 1**.

Data `forecast` disimpan sebagai snapshot untuk tahap inference berikutnya dan belum dipakai di prapemrosesan.

### Format Data Mentah

Lokasi: `data/raw/data_YYYYMMDD_HHMMSS_<sumber>.csv`

| Sumber   | Kolom |
|----------|-------|
| flood    | `time` (tanggal), `river_discharge` (m³/s), `ingested_at` |
| weather  | `time` (per jam, WIB), `precipitation` (mm), `soil_moisture_0_to_7cm` (m³/m³), `ingested_at` |
| forecast | `time` (per jam, WIB), `precipitation` (mm), `ingested_at` |

Catatan: Flood API tidak menyediakan parameter timezone sehingga tanggal pada data flood mengikuti GMT. Weather dan forecast diminta dalam zona waktu `Asia/Jakarta`.

---

## 2. Prapemrosesan (`src/preprocess.py`)

```bash
python src/preprocess.py
```

Argumen opsional:

- `--raw-dir PATH`: folder sumber file raw (default `data/raw`)
- `--processed-dir PATH`: folder tujuan dataset (default `data/processed`)

Tahapan:

1. Muat dan gabungkan semua file raw per sumber (`flood` dan `weather`).
2. Buang timestamp ganda. Data dengan `ingested_at` terbaru dipertahankan.
3. Validasi nilai fisik. Nilai di luar batas diubah menjadi kosong: `river_discharge` dan `precipitation` minimal 0, `soil_moisture` antara 0 dan 1.
4. Agregasi weather ke harian: hujan dijumlah, kelembapan tanah dirata-rata. Hari yang tidak lengkap 24 jam dibuang.
5. Gabungkan data debit dan cuaca berdasarkan tanggal (`inner join`).
6. Lengkapi indeks harian, lalu isi nilai kosong: interpolasi waktu untuk celah sampai 3 hari pada `river_discharge` dan `soil_moisture`, celah lebih panjang dibiarkan kosong, dan `precipitation` kosong diisi 0.
7. Bentuk fitur:
   - Lag debit: `river_discharge_lag1`, `river_discharge_lag2`, `river_discharge_lag3`
   - Rolling hujan: `precipitation_roll3`, `precipitation_roll7`
   - Rolling kelembapan tanah: `soil_moisture_roll3`
   - Musiman: `month`, `is_rainy_season` (bernilai 1 untuk bulan 11, 12, 1, 2, 3)
8. Buang baris yang masih memiliki nilai kosong.
9. Tampilkan peringatan jika jumlah baris di bawah 500 (batas yang disarankan untuk versioning DVC).
10. Simpan ke `data/processed/dataset_YYYYMMDD_HHMMSS.csv` tanpa menimpa file lama.

Prapemrosesan disesuaikan dengan domain data deret waktu hidrologi, bukan teks sehingga tahapannya berupa pembersihan nilai, agregasi temporal, dan rekayasa fitur waktu.

### Format Dataset Akhir

```
date, river_discharge, precipitation, soil_moisture,
river_discharge_lag1, river_discharge_lag2, river_discharge_lag3,
precipitation_roll3, precipitation_roll7, soil_moisture_roll3,
month, is_rainy_season
```

---

## 3. Versioning Data dengan DVC

### Remote storage: MinIO

Remote DVC berupa bucket MinIO (S3-compatible) dengan konfigurasi:

- Bucket: `dvc-data`, path `dvcstore`
- Endpoint: `http://localhost:9000` (server MinIO berjalan di dalam Codespaces)
- Kredensial **tidak** disimpan di Git. Kredensial berada di `.dvc/config.local` yang diabaikan Git secara otomatis.

Edisi komunitas MinIO kini hanya dibagikan sebagai source code sehingga server dibangun dengan Go. Langkah pemasangan:

```bash
GO_VERSION=$(curl -s 'https://go.dev/VERSION?m=text' | head -1)
curl -fL -o /tmp/go.tgz "https://go.dev/dl/${GO_VERSION}.linux-amd64.tar.gz"
mkdir -p /workspaces/go-sdk
tar -C /workspaces/go-sdk -xzf /tmp/go.tgz
export PATH=/workspaces/go-sdk/go/bin:$PATH
export GOPATH=/workspaces/gopath
go install github.com/minio/minio@latest
```

Menjalankan server (ganti `<user>` dan `<password>`, password minimal 8 karakter):

```bash
export MINIO_ROOT_USER=<user>
export MINIO_ROOT_PASSWORD=<password>
mkdir -p /workspaces/minio-data
nohup /workspaces/gopath/bin/minio server /workspaces/minio-data --console-address ":9001" > /workspaces/minio.log 2>&1 &
curl -i http://localhost:9000/minio/health/live
```

Respons `HTTP/1.1 200 OK` menandakan server siap.

Membuat bucket :

```bash
pip install boto3
python -c "import os, boto3; s3 = boto3.client('s3', endpoint_url='http://localhost:9000', aws_access_key_id=os.environ['MINIO_ROOT_USER'], aws_secret_access_key=os.environ['MINIO_ROOT_PASSWORD'], region_name='us-east-1'); s3.create_bucket(Bucket='dvc-data')"
```

Konfigurasi DVC :

```bash
dvc init
dvc remote add -d minio s3://dvc-data/dvcstore
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify --local minio access_key_id "$MINIO_ROOT_USER"
dvc remote modify --local minio secret_access_key "$MINIO_ROOT_PASSWORD"
```

### Catatan operasional MinIO

- Server MinIO ikut berhenti ketika Codespaces berhenti. Sebelum `dvc push` atau `dvc pull`, jalankan ulang perintah `nohup ... minio server` di atas dengan kredensial yang sama.
- Data tetap tersimpan di `/workspaces/minio-data` selama Codespaces tidak dihapus.
- Jika kredensial terlupa, nilainya ada di `.dvc/config.local`.
- Jangan mengubah visibilitas port 9000 dan 9001 menjadi public.

### Alur versi 1 (pelacakan dataset awal)

File yang sebelumnya dilacak Git perlu dilepas terlebih dahulu:

```bash
git rm -r --cached data/raw data/processed
dvc add data/raw data/processed
git add data/raw.dvc data/processed.dvc data/.gitignore
git commit -m "feat(data): track dataset v1 with dvc"
git tag -a data-v1 -m "dataset version 1"
dvc push
```

### Alur versi baru (simulasi Continual Learning)

```bash
python src/ingest_data.py
python src/preprocess.py
dvc status
dvc add data/raw data/processed
git add data/raw.dvc data/processed.dvc
git commit -m "feat(data): track dataset v2 with new ingested data"
git tag -a data-v2 -m "dataset version 2"
dvc push
git push origin <nama-branch> --tags
```

Konvensi: setiap versi data diberi tag Git `data-vN` dan commit penunjuk `.dvc` memakai prefix `feat(data):`.

### Audit dan perbandingan versi

```bash
dvc status -c                    
dvc diff data-v1 data-v2         
dvc diff data-v1 data-v2 --show-hash
git --no-pager diff data-v1 data-v2 -- data/raw.dvc data/processed.dvc
git log --oneline --decorate
```

Ringkasan versi yang tercatat:

| Tag | Isi |
|-----|-----|
| `data-v1` | Backfill 2024-01-01 sampai 2026-09-22, dataset akhir 990 baris |
| `data-v2` | Tambahan data 2026-09-15 sampai 2026-10-05, yaitu 3 file raw dan 1 dataset baru |

Hasil `dvc diff data-v1 data-v2 --show-hash` menunjukkan hash folder berubah: `data/raw` dari `3eba40cc` menjadi `18c52e85` dan `data/processed` dari `843b1e87` menjadi `9aaa854a`.

### Memulihkan versi data tertentu

```bash
git checkout data-v1
dvc checkout
git checkout <nama-branch>
dvc checkout
```

Setelah `git checkout` ke sebuah tag, Git menampilkan pesan `detached HEAD`. Jangan membuat commit pada kondisi tersebut.

### Mereproduksi di lingkungan lain

Remote MinIO berjalan di Codespaces pemilik repositori sehingga `dvc pull` dari lingkungan lain tidak dapat menjangkaunya. Untuk membangun ulang data, jalankan pipeline di bagian 1 dan 2. Untuk memakai remote sendiri, jalankan MinIO dengan langkah di atas lalu ubah `endpointurl` dengan `dvc remote modify`.

---

## 4. Penjadwalan Periodik

Pengambilan berkala dapat dijadwalkan dengan **cron** di server Linux, misalnya setiap hari pukul 07.00:

```bash
0 7 * * * cd /path/ke/MLOps-PrediksiDebitSungai && python src/ingest_data.py && python src/preprocess.py
```

Karena memakai `&&`, jika `ingest_data.py` mengembalikan exit code 1 (ada sumber yang gagal), `preprocess.py` tidak dijalankan. Hal ini disengaja agar dataset tidak dibentuk dari data yang tidak lengkap. Ganti `&&` dengan `;` jika prapemrosesan tetap ingin dijalankan.

Di **GitHub Codespaces**, simulasi periodik dilakukan dengan menjalankan ulang kedua skrip secara manual:

```bash
python src/ingest_data.py && python src/preprocess.py
```