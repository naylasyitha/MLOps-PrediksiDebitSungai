import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DVC_POINTER = ROOT_DIR / "data" / "processed.dvc"

TARGET = "river_discharge"
DATE_COLUMN = "date"
LAG_COLUMN = "river_discharge_lag1"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
HIGH_FLOW_QUANTILE = 0.90

DEFAULT_EXPERIMENT = "prediksi-debit-sungai"
HYPERPARAMETERS = {
    "baseline": (),
    "rf": ("n_estimators", "max_depth", "min_samples_leaf"),
    "gbr": ("n_estimators", "learning_rate", "max_depth"),
}
GBR_DEFAULT_DEPTH = 3
SERIALIZATION_FORMAT = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE


class PersistenceBaseline(RegressorMixin, BaseEstimator):
    """Baseline: debit hari ini diprediksi sama dengan debit kemarin."""

    def fit(self, features, target):
        """Tidak ada yang dipelajari, hanya memenuhi antarmuka sklearn."""
        return self

    def predict(self, features):
        """Kembalikan debit kemarin sebagai prediksi."""
        return features[LAG_COLUMN].to_numpy()


def find_latest_dataset(processed_dir):
    """Pilih file dataset terbaru berdasarkan timestamp pada nama file."""
    candidates = sorted(processed_dir.glob("dataset_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Tidak ada dataset di {processed_dir}")
    return candidates[-1]


def load_dataset(csv_path):
    """Baca dataset dan urutkan berdasarkan tanggal."""
    frame = pd.read_csv(csv_path, parse_dates=[DATE_COLUMN])
    return frame.sort_values(DATE_COLUMN).reset_index(drop=True)


def split_by_time(frame):
    """Bagi data berurutan waktu menjadi train, validasi, dan test."""
    total = len(frame)
    train_end = int(total * TRAIN_RATIO)
    validation_end = int(total * (TRAIN_RATIO + VALIDATION_RATIO))
    return (
        frame.iloc[:train_end],
        frame.iloc[train_end:validation_end],
        frame.iloc[validation_end:],
    )


def split_features_target(frame):
    """Pisahkan fitur dari target, tanggal dan target bukan fitur."""
    feature_columns = [
        column
        for column in frame.columns
        if column not in (DATE_COLUMN, TARGET)
    ]
    return frame[feature_columns], frame[TARGET]


def nash_sutcliffe(observed, predicted):
    """Hitung Nash-Sutcliffe Efficiency."""
    residual = np.sum((observed - predicted) ** 2)
    variance = np.sum((observed - np.mean(observed)) ** 2)
    return 1.0 - residual / variance


def rmse(observed, predicted):
    """Hitung root mean squared error."""
    return float(np.sqrt(mean_squared_error(observed, predicted)))


def evaluate(observed, predicted, high_flow_threshold):
    """Hitung metrik regresi, termasuk galat pada debit tinggi."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    metrics = {
        "rmse": rmse(observed, predicted),
        "mae": float(mean_absolute_error(observed, predicted)),
        "nse": float(nash_sutcliffe(observed, predicted)),
    }
    high_flow = observed >= high_flow_threshold
    if high_flow.any():
        metrics["rmse_high_flow"] = rmse(
            observed[high_flow], predicted[high_flow]
        )
    return metrics


def apply_model_defaults(args):
    """Lengkapi nilai bawaan hyperparameter yang bergantung pada model."""
    if args.model == "gbr" and args.max_depth is None:
        args.max_depth = GBR_DEFAULT_DEPTH
    return args


def build_model(args):
    """Buat model sesuai pilihan argumen."""
    if args.model == "baseline":
        return PersistenceBaseline()
    if args.model == "rf":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
        )
    return GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.seed,
    )


def read_dvc_hash(pointer_path):
    """Ambil hash md5 dari file penunjuk DVC, atau unknown."""
    if not pointer_path.exists():
        return "unknown"
    match = re.search(r"md5:\s*(\S+)", pointer_path.read_text())
    return match.group(1) if match else "unknown"


def read_data_tag():
    """Ambil tag Git data-v* terdekat, atau unknown."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--match", "data-v*", "--abbrev=0"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def run_experiment(args):
    """Latih model, hitung metrik, dan catat semuanya ke MLflow."""
    data_path = args.data_path or find_latest_dataset(PROCESSED_DIR)
    frame = load_dataset(data_path)
    train, validation, test = split_by_time(frame)
    splits = {
        "train": split_features_target(train),
        "val": split_features_target(validation),
        "test": split_features_target(test),
    }
    x_train, y_train = splits["train"]
    threshold = float(y_train.quantile(HIGH_FLOW_QUANTILE))

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name or args.model) as run:
        params = {"model": args.model, "seed": args.seed}
        for name in HYPERPARAMETERS[args.model]:
            params[name] = getattr(args, name)
        params.update(
            {
                "n_features": x_train.shape[1],
                "rows_train": len(train),
                "rows_val": len(validation),
                "rows_test": len(test),
                "high_flow_threshold": round(threshold, 2),
            }
        )
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "data_file": Path(data_path).name,
                "data_tag": read_data_tag(),
                "dvc_processed_md5": read_dvc_hash(DVC_POINTER),
            }
        )

        model = build_model(args)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        mlflow.log_metric("train_seconds", time.perf_counter() - started)

        for split_name, (features, observed) in splits.items():
            metrics = evaluate(observed, model.predict(features), threshold)
            mlflow.log_metrics(
                {
                    f"{split_name}_{key}": value
                    for key, value in metrics.items()
                }
            )

        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=x_train.head(3),
            serialization_format=SERIALIZATION_FORMAT,
        )
    return run.info.run_id


def parse_args():
    """Baca argumen command line."""
    parser = argparse.ArgumentParser(
        description="Latih model prediksi debit sungai dan catat ke MLflow."
    )
    parser.add_argument(
        "--model",
        choices=sorted(HYPERPARAMETERS),
        default="rf",
        help="Jenis model. Default: %(default)s",
    )
    parser.add_argument("--run-name", default=None, help="Nama run di MLflow.")
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT,
        help="Nama eksperimen. Default: %(default)s",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Dataset CSV. Default: file terbaru di data/processed.",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Kedalaman pohon. Kosongkan untuk tanpa batas (rf).",
    )
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """Jalankan satu eksperimen dan kembalikan exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = apply_model_defaults(parse_args())
    logging.info("Tracking URI: %s", mlflow.get_tracking_uri())
    run_id = run_experiment(args)
    logging.info("Run selesai: %s", run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())