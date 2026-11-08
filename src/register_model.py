
import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.evaluate_model import validate_model


def register_staging(evaluation_path, config_path, output_path):
    # Membaca hasil evaluasi dan konfigurasi baseline
    evaluation = json.loads(
        Path(evaluation_path).read_text(encoding="utf-8")
    )
    config = json.loads(
        Path(config_path).read_text(encoding="utf-8")
    )

    run_id = evaluation.get("run_id")

    if not run_id:
        raise ValueError("Run ID tidak ditemukan.")

    # Model hanya boleh diregistrasikan jika evaluasi lolos
    if (
        evaluation.get("status") != "PASSED"
        or evaluation.get("eligible_for_registration") is not True
    ):
        raise ValueError("Model belum lolos evaluasi.")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI belum dikonfigurasi.")

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    # Mengambil run asli dari MLflow
    run = client.get_run(run_id)

    # Validasi ulang menggunakan metrik asli MLflow
    validation = validate_model(
        metrics=run.data.metrics,
        tags=run.data.tags,
        config=config,
    )

    if not validation["eligible_for_registration"]:
        raise ValueError("Validasi ulang model gagal.")

    if validation["new_rmse"] != evaluation.get("new_rmse"):
        raise ValueError(
            "Metrik evaluasi tidak sesuai dengan MLflow."
        )

    print("=== LK08 REGISTRY VALIDATION ===")
    print("Run ID:", run_id)
    print("Validation RMSE:", validation["new_rmse"])
    print("Threshold RMSE:", validation["threshold_rmse"])
    print("Validation Status: PASSED")

    # Mencari logged model berdasarkan eksperimen dan Run ID
    logged_models = client.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string="name = 'model'",
    )

    candidates = [
        model
        for model in logged_models
        if model.source_run_id == run_id
    ]

    print("Logged models ditemukan:", len(candidates))

    if len(candidates) == 0:
        raise ValueError(
            "Model hasil training tidak ditemukan pada Run ID ini."
        )

    if len(candidates) > 1:
        raise ValueError(
            "Terdapat lebih dari satu model untuk Run ID ini."
        )

    # Mengambil Model ID MLflow 3
    logged_model = candidates[0]
    model_id = logged_model.model_id
    model_uri = f"models:/{model_id}"

    model_name = config["model_name"]

    print("Model ID:", model_id)
    print("Model URI:", model_uri)

    # Registrasi hanya dilakukan setelah validasi berhasil
    version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "deployment_status": "staging",
            "validation_status": "PASSED",
            "source_run_id": run_id,
            "validation_rmse": str(validation["new_rmse"]),
            "dvc_processed_md5": validation["dvc_processed_md5"],
        },
    )

    # Menetapkan alias staging tanpa mengubah champion
    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=version.version,
    )

    # Menyimpan hasil registrasi
    result = {
        "status": "REGISTERED",
        "model_name": model_name,
        "model_version": str(version.version),
        "model_id": model_id,
        "run_id": run_id,
        "deployment_status": "staging",
        "validation_status": "PASSED",
        "validation_rmse": validation["new_rmse"],
        "model_uri": model_uri,
    }

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("=== LK08 AUTO REGISTRY ===")
    print("Model:", model_name)
    print("Version:", version.version)
    print("Run ID:", run_id)
    print("Deployment Status: staging")
    print("Registration: SUCCESS")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Registrasi otomatis model LK08 ke MLflow Staging."
    )

    parser.add_argument(
        "--evaluation-file",
        required=True,
    )

    parser.add_argument(
        "--config",
        default="docs/lk08_evaluation_config.json",
    )

    parser.add_argument(
        "--output",
        default="registration-result.json",
    )

    args = parser.parse_args()

    try:
        register_staging(
            evaluation_path=args.evaluation_file,
            config_path=args.config,
            output_path=args.output,
        )
        return 0

    except ValueError as error:
        print(
            "AUTO REGISTRY VALIDATION FAILED:",
            str(error),
            file=sys.stderr,
        )
        return 1

    except Exception as error:
        print(
            "AUTO REGISTRY FAILED:",
            type(error).__name__,
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
