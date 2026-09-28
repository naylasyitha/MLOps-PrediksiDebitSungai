"""Data ingestion untuk sistem prediksi debit sungai di Jakarta.

Skrip ini mengambil data dari tiga endpoint Open-Meteo, yaitu flood,
weather historis, dan forecast lalu menyimpannya sebagai file CSV 
bertimestamp di ``data/raw/``.

Run pertama melakukan backfill mulai dari ``--start-date``. Run
berikutnya hanya mengambil data sejak tanggal terakhir yang sudah ada 
ditambah overlap beberapa hari untuk menangkap revisi nilai.

Contoh:
    python src/ingest_data.py
    python src/ingest_data.py --start-date 2025-01-01
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

LATITUDE = -6.2146
LONGITUDE = 106.8451
TIMEZONE = "Asia/Jakarta"

DEFAULT_START_DATE = "2024-01-01"
OVERLAP_DAYS = 7
ARCHIVE_DELAY_DAYS = 6
FORECAST_DAYS = 7

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
BACKOFF_SECONDS = 2
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCES = {
    "flood": {
        "url": "https://flood-api.open-meteo.com/v1/flood",
        "block": "daily",
        "params": {"daily": "river_discharge"},
        "use_date_window": True,
    },
    "weather": {
        "url": "https://archive-api.open-meteo.com/v1/archive",
        "block": "hourly",
        "params": {
            "hourly": "precipitation,soil_moisture_0_to_7cm",
            "timezone": TIMEZONE,
        },
        "use_date_window": True,
    },
    "forecast": {
        "url": "https://api.open-meteo.com/v1/forecast",
        "block": "hourly",
        "params": {
            "hourly": "precipitation",
            "forecast_days": FORECAST_DAYS,
            "timezone": TIMEZONE,
        },
        "use_date_window": False,
    },
}


class IngestionError(Exception):
    """Error yang terjadi saat proses pengambilan data."""


def fetch_json(url, params):
    """Ambil JSON dari API dengan timeout dan retry.

    Error koneksi, timeout, dan status sementara diulang
    hingga MAX_RETRIES kali dengan exponential backoff.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except requests.RequestException as error:
            last_error = error
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as error:
                    raise IngestionError("Respons bukan JSON valid") from error
            if response.status_code not in RETRYABLE_STATUS_CODES:
                raise IngestionError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
            last_error = f"HTTP {response.status_code}"

        logging.warning(
            "Percobaan %d/%d gagal: %s", attempt, MAX_RETRIES, last_error
        )
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))

    raise IngestionError(
        f"Gagal setelah {MAX_RETRIES} percobaan: {last_error}"
    )


def find_last_date(output_dir, source_name):
    """Cari tanggal terakhir pada file raw yang sudah ada.

    Mengembalikan None jika belum ada file untuk sumber tersebut.
    """
    last_date = None
    for csv_path in sorted(output_dir.glob(f"data_*_{source_name}.csv")):
        times = pd.read_csv(csv_path, usecols=["time"])["time"]
        if times.empty:
            continue
        file_last_date = pd.to_datetime(times).max().date()
        if last_date is None or file_last_date > last_date:
            last_date = file_last_date
    return last_date


def build_date_window(last_date, start_date, end_date):
    """Tentukan rentang tanggal yang akan diambil.

    Tanpa data sebelumnya, pakai start_date (backfill). Jika sudah ada,
    mulai dari tanggal terakhir dikurangi OVERLAP_DAYS.
    """
    if last_date is not None:
        start_date = last_date - timedelta(days=OVERLAP_DAYS)
    return min(start_date, end_date), end_date


def payload_to_frame(payload, block):
    """Ubah blok JSON (daily atau hourly) menjadi DataFrame mentah."""
    data = payload.get(block)
    if not data or not data.get("time"):
        raise IngestionError(f"Respons tidak memuat data pada blok '{block}'")

    frame = pd.DataFrame(data)
    if frame.drop(columns="time").isna().all().all():
        raise IngestionError("Semua nilai pada respons kosong")
    return frame


def save_frame(frame, output_dir, run_timestamp, source_name):
    """Simpan DataFrame ke CSV baru tanpa menimpa file yang ada."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"data_{run_timestamp}_{source_name}.csv"
    if final_path.exists():
        raise IngestionError(f"File sudah ada: {final_path.name}")

    temp_path = final_path.with_name(final_path.name + ".tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(final_path)
    return final_path


def ingest_source(
    source_name, output_dir, run_timestamp, start_date, end_date
):
    """Ambil satu sumber data dan simpan sebagai file raw baru."""
    config = SOURCES[source_name]
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        **config["params"],
    }

    if config["use_date_window"]:
        last_date = find_last_date(output_dir, source_name)
        window_start, window_end = build_date_window(
            last_date, start_date, end_date
        )
        params["start_date"] = window_start.isoformat()
        params["end_date"] = window_end.isoformat()
        logging.info(
            "Sumber '%s': mengambil %s sampai %s",
            source_name,
            params["start_date"],
            params["end_date"],
        )
    else:
        logging.info("Sumber '%s': mengambil snapshot forecast", source_name)

    payload = fetch_json(config["url"], params)
    frame = payload_to_frame(payload, config["block"])
    frame["ingested_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    logging.info("Sumber '%s': %d baris diterima", source_name, len(frame))
    return save_frame(frame, output_dir, run_timestamp, source_name)


def parse_args():
    """Baca argumen command line."""
    parser = argparse.ArgumentParser(
        description="Ambil data Open-Meteo dan simpan sebagai CSV mentah."
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date.fromisoformat(DEFAULT_START_DATE),
        help="Tanggal awal backfill (YYYY-MM-DD), hanya dipakai pada run "
        "pertama. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder tujuan file raw. Default: %(default)s",
    )
    return parser.parse_args()


def main():
    """Jalankan ingestion untuk semua sumber. Kembalikan exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    today = datetime.now(timezone.utc).date()
    end_date = today - timedelta(days=ARCHIVE_DELAY_DAYS)

    failed_sources = []
    for source_name in SOURCES:
        try:
            saved_path = ingest_source(
                source_name,
                args.output_dir,
                run_timestamp,
                args.start_date,
                end_date,
            )
        except IngestionError as error:
            logging.error("Sumber '%s' gagal: %s", source_name, error)
            failed_sources.append(source_name)
        else:
            logging.info("Sumber '%s' tersimpan: %s", source_name, saved_path)

    if failed_sources:
        logging.error("Sumber yang gagal: %s", ", ".join(failed_sources))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())