import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

HOURS_PER_DAY = 24
MAX_INTERPOLATION_GAP_DAYS = 3
MIN_ROWS_FOR_DVC = 500

LAG_DAYS = (1, 2, 3)
PRECIPITATION_WINDOWS = (3, 7)
SOIL_MOISTURE_WINDOW = 3
RAINY_SEASON_MONTHS = (11, 12, 1, 2, 3)

VALID_RANGES = {
    "river_discharge": (0.0, float("inf")),
    "precipitation": (0.0, float("inf")),
    "soil_moisture": (0.0, 1.0),
}


class PreprocessingError(Exception):
    pass


def load_raw(raw_dir, source_name):
    """Gabungkan semua file raw milik satu sumber."""
    csv_paths = sorted(raw_dir.glob(f"data_*_{source_name}.csv"))
    if not csv_paths:
        raise PreprocessingError(f"Tidak ada file raw untuk '{source_name}'")
    frames = [pd.read_csv(csv_path) for csv_path in csv_paths]
    logging.info("Sumber '%s': %d file dimuat", source_name, len(frames))
    return pd.concat(frames, ignore_index=True)


def keep_latest_records(frame):
    """Untuk timestamp ganda, pertahankan data dengan ingested_at terbaru."""
    ordered = frame.sort_values(["time", "ingested_at"])
    unique = ordered.drop_duplicates(subset="time", keep="last")
    logging.info("Duplikat dibuang: %d baris", len(frame) - len(unique))
    return unique.drop(columns="ingested_at")


def remove_invalid_values(frame):
    """Ubah nilai di luar batas fisik menjadi NaN."""
    for column, (lower, upper) in VALID_RANGES.items():
        if column not in frame.columns:
            continue
        invalid = (frame[column] < lower) | (frame[column] > upper)
        logging.info("Nilai tidak valid pada %s: %d", column, invalid.sum())
        frame.loc[invalid, column] = float("nan")
    return frame


def prepare_flood(flood):
    """Rapikan data debit sungai harian dengan indeks tanggal."""
    frame = remove_invalid_values(keep_latest_records(flood))
    frame["date"] = pd.to_datetime(frame["time"])
    return frame.drop(columns="time").set_index("date")


def aggregate_weather_daily(weather):
    """Agregasi data hourly ke harian dan buang hari yang tidak lengkap."""
    frame = keep_latest_records(weather)
    frame = frame.rename(columns={"soil_moisture_0_to_7cm": "soil_moisture"})
    frame = remove_invalid_values(frame)
    frame = frame.dropna(subset=["precipitation", "soil_moisture"])
    frame["date"] = pd.to_datetime(frame["time"]).dt.normalize()

    daily = frame.groupby("date").agg(
        precipitation=("precipitation", "sum"),
        soil_moisture=("soil_moisture", "mean"),
        hours=("time", "count"),
    )
    incomplete = daily["hours"] < HOURS_PER_DAY
    logging.info("Hari tidak lengkap dibuang: %d", incomplete.sum())
    return daily.loc[~incomplete].drop(columns="hours")


def merge_sources(flood_daily, weather_daily):
    """Gabungkan data debit dan cuaca berdasarkan tanggal."""
    merged = flood_daily.join(weather_daily, how="inner")
    if merged.empty:
        raise PreprocessingError("Tidak ada tanggal yang cocok antar sumber")
    logging.info("Hasil penggabungan: %d baris", len(merged))
    return merged


def interpolate_short_gaps(series):
    """Interpolasi hanya untuk celah pendek, celah panjang dibiarkan kosong."""
    filled = series.interpolate(method="time", limit_area="inside")
    is_missing = series.isna()
    gap_id = (~is_missing).cumsum()
    gap_length = is_missing.groupby(gap_id).transform("sum")
    too_long = is_missing & (gap_length > MAX_INTERPOLATION_GAP_DAYS)
    return filled.where(~too_long)


def fill_missing_values(frame):
    """Lengkapi indeks harian lalu isi nilai kosong yang pendek."""
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    frame = frame.reindex(full_index)
    frame.index.name = "date"
    logging.info("Nilai kosong sebelum diisi: %d", frame.isna().sum().sum())

    for column in ("river_discharge", "soil_moisture"):
        frame[column] = interpolate_short_gaps(frame[column])
    frame["precipitation"] = frame["precipitation"].fillna(0.0)
    logging.info("Nilai kosong setelah diisi: %d", frame.isna().sum().sum())
    return frame


def add_features(frame):
    """Tambahkan fitur lag, rolling, dan musiman."""
    for lag in LAG_DAYS:
        frame[f"river_discharge_lag{lag}"] = frame["river_discharge"].shift(
            lag
        )
    for window in PRECIPITATION_WINDOWS:
        frame[f"precipitation_roll{window}"] = (
            frame["precipitation"].rolling(window).sum()
        )
    frame[f"soil_moisture_roll{SOIL_MOISTURE_WINDOW}"] = (
        frame["soil_moisture"].rolling(SOIL_MOISTURE_WINDOW).mean()
    )
    frame["month"] = frame.index.month
    frame["is_rainy_season"] = frame["month"].isin(RAINY_SEASON_MONTHS)
    frame["is_rainy_season"] = frame["is_rainy_season"].astype(int)
    return frame


def save_dataset(frame, processed_dir, run_timestamp):
    """Simpan dataset akhir ke CSV baru tanpa menimpa file lama."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / f"dataset_{run_timestamp}.csv"
    if output_path.exists():
        raise PreprocessingError(f"File sudah ada: {output_path.name}")
    frame.to_csv(output_path)
    return output_path


def run_pipeline(raw_dir, processed_dir):
    """Jalankan seluruh tahap prapemrosesan dan simpan hasilnya."""
    flood_daily = prepare_flood(load_raw(raw_dir, "flood"))
    weather_daily = aggregate_weather_daily(load_raw(raw_dir, "weather"))

    frame = merge_sources(flood_daily, weather_daily)
    frame = fill_missing_values(frame)
    frame = add_features(frame)

    row_count = len(frame)
    frame = frame.dropna()
    logging.info(
        "Baris dibuang karena fitur tidak lengkap: %d", row_count - len(frame)
    )
    if len(frame) < MIN_ROWS_FOR_DVC:
        logging.warning(
            "Jumlah baris %d di bawah %d yang disarankan untuk LK-05",
            len(frame),
            MIN_ROWS_FOR_DVC,
        )

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return save_dataset(frame, processed_dir, run_timestamp)


def parse_args():
    """Baca argumen command line."""
    parser = argparse.ArgumentParser(
        description="Bersihkan data raw dan bentuk dataset siap pakai."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Folder sumber file raw. Default: %(default)s",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Folder tujuan dataset. Default: %(default)s",
    )
    return parser.parse_args()


def main():
    """Jalankan prapemrosesan dan kembalikan exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    try:
        output_path = run_pipeline(args.raw_dir, args.processed_dir)
    except PreprocessingError as error:
        logging.error("Prapemrosesan gagal: %s", error)
        return 1
    logging.info("Dataset tersimpan: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())