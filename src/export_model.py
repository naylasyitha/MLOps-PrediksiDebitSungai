
from pathlib import Path
import shutil

import mlflow.artifacts
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "debit-sungai-regressor"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

MODEL_DIR = ROOT / "models" / "champion"
TEMP_DIR = ROOT / "models" / "_temp_export"


def main():
    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion"
    )

    if champion.tags.get("deployment_status") != "production":
        raise RuntimeError(
            "Model Champion belum berstatus production."
        )

    print("Model Name:", MODEL_NAME)
    print("Champion Version:", champion.version)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = mlflow.artifacts.download_artifacts(
        artifact_uri=MODEL_URI,
        dst_path=str(TEMP_DIR),
    )

    downloaded = Path(downloaded_path)

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    shutil.copytree(downloaded, MODEL_DIR)
    shutil.rmtree(TEMP_DIR)

    if not (MODEL_DIR / "MLmodel").exists():
        raise RuntimeError("Artefak MLflow tidak lengkap.")

    print("Model berhasil diekspor!")
    print("Lokasi:", MODEL_DIR)


if __name__ == "__main__":
    main()
