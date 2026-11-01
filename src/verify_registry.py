
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "debit-sungai-regressor"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

DATA_DIR = ROOT / "data" / "processed"


def main():
    # 1. Ambil informasi model Champion
    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion"
    )

    # 2. Cari dataset hasil preprocessing
    datasets = sorted(DATA_DIR.glob("dataset_*.csv"))

    if not datasets:
        raise FileNotFoundError(
            "Dataset tidak ditemukan di data/processed/"
        )

    dataset_path = datasets[-1]
    data = pd.read_csv(dataset_path)

    # 3. Siapkan fitur input
    features = data.drop(
        columns=["date", "river_discharge"]
    )

    sample = features.iloc[[0]]

    # 4. Load model dari MLflow Model Registry
    model = mlflow.pyfunc.load_model(MODEL_URI)

    # 5. Lakukan prediksi
    prediction = model.predict(sample)

    # 6. Tampilkan hasil
    print("=== VERIFIKASI INFERENSI LK-07 ===")
    print("Model Name       :", MODEL_NAME)
    print("Model Version    :", champion.version)
    print("Deployment Status:", champion.tags.get(
        "deployment_status"
    ))
    print("Model URI        :", MODEL_URI)
    print("Dataset          :", dataset_path.name)
    print("Jumlah Fitur     :", sample.shape[1])
    print("Prediksi Debit   :", float(prediction[0]), "m3/s")
    print("Status           : SUCCESS")


if __name__ == "__main__":
    main()
