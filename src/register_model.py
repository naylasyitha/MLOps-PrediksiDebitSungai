
import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.evaluate_model import validate_model


def register_staging(evaluation_path, config_path, output_path):
    evaluation = json.loads(Path(evaluation_path).read_text())
    config = json.loads(Path(config_path).read_text())

    run_id = evaluation.get("run_id")
    if not run_id:
        raise ValueError("Run ID tidak ditemukan.")

    if (
        evaluation.get("status") != "PASSED"
        or evaluation.get("eligible_for_registration") is not True
    ):
        raise ValueError("Model belum lolos evaluasi.")

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    # Validasi ulang menggunakan metrik asli MLflow
    validation = validate_model(
        run.data.metrics,
        run.data.tags,
        config,
    )

    if not validation["eligible_for_registration"]:
        raise ValueError("Validasi ulang model gagal.")

    if validation["new_rmse"] != evaluation.get("new_rmse"):
        raise ValueError("Metrik evaluasi tidak sesuai.")

    # MLflow 3 menggunakan Model ID
    logged_models = mlflow.search_logged_models(
        filter_string=f"source_run_id = '{run_id}'",
        output_format="list",
    )

    candidates = [
        model for model in logged_models
        if model.name == "model"
    ]

    if len(candidates) != 1:
        raise ValueError(
            "Model hasil training tidak ditemukan atau ambigu."
        )

    model_id = candidates[0].model_id
    model_uri = f"models:/{model_id}"
    model_name = config["model_name"]

    # Registrasi dilakukan hanya setelah validasi lolos
    version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "deployment_status": "staging",
            "validation_status": "PASSED",
            "source_run_id": run_id,
            "validation_rmse": str(validation["new_rmse"]),
        },
    )

    # Alias staging tanpa mengubah alias champion
    client.set_registered_model_alias(
        model_name,
        "staging",
        version.version,
    )

    result = {
        "status": "REGISTERED",
        "model_name": model_name,
        "model_version": str(version.version),
        "model_id": model_id,
        "run_id": run_id,
        "deployment_status": "staging",
        "validation_status": "PASSED",
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print("=== LK08 AUTO REGISTRY ===")
    print("Model:", model_name)
    print("Version:", version.version)
    print("Run ID:", run_id)
    print("Deployment Status: staging")
    print("Registration: SUCCESS")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-file", required=True)
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
            args.evaluation_file,
            args.config,
            args.output,
        )
        return 0
    except Exception as error:
        print(
            "AUTO REGISTRY FAILED:",
            type(error).__name__,
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
