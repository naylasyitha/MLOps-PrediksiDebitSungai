
import argparse
import json
import math
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT_DIR / "docs" / "lk08_evaluation_config.json"


def validate_model(metrics, tags, config):
    """Validasi performa dan versi dataset model baru."""
    metric_name = config["metric_name"]
    threshold = float(config["max_val_rmse"])

    if metric_name not in metrics:
        raise ValueError(f"Metrik {metric_name} tidak ditemukan.")

    new_rmse = float(metrics[metric_name])

    if not math.isfinite(new_rmse) or new_rmse < 0:
        raise ValueError("Nilai RMSE model baru tidak valid.")

    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold RMSE tidak valid.")

    if tags.get("data_file") != config["dataset_file"]:
        raise ValueError(
            "Dataset model baru berbeda dari baseline LK07."
        )

    if tags.get("dvc_processed_md5") != config["dvc_processed_md5"]:
        raise ValueError(
            "Versi dataset DVC berbeda dari baseline LK07."
        )

    passed = new_rmse <= threshold

    return {
        "status": "PASSED" if passed else "FAILED",
        "metric_name": metric_name,
        "new_rmse": new_rmse,
        "threshold_rmse": threshold,
        "dataset_file": tags["data_file"],
        "dvc_processed_md5": tags["dvc_processed_md5"],
        "model_name": config["model_name"],
        "champion_version": config["champion_version"],
        "eligible_for_registration": passed,
    }


def evaluate_run(run_id, tracking_uri, config_path, output_path):
    """Ambil metrik dari MLflow dan buat keputusan validasi."""
    config = json.loads(Path(config_path).read_text())

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    result = validate_model(
        metrics=run.data.metrics,
        tags=run.data.tags,
        config=config,
    )

    result["run_id"] = run_id

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print("=== LK08 MODEL EVALUATION ===")
    print("Run ID:", run_id)
    print("Model:", result["model_name"])
    print("Validation RMSE:", result["new_rmse"])
    print("Threshold RMSE:", result["threshold_rmse"])
    print("Dataset:", result["dataset_file"])
    print("Status:", result["status"])

    return result["eligible_for_registration"]


def main():
    parser = argparse.ArgumentParser(
        description="Validasi otomatis model hasil retraining LK08."
    )

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output",
        default="evaluation-result.json",
    )

    args = parser.parse_args()

    try:
        passed = evaluate_run(
            run_id=args.run_id,
            tracking_uri=args.tracking_uri,
            config_path=args.config,
            output_path=args.output,
        )
        return 0 if passed else 1

    except Exception as error:
        print(f"MODEL EVALUATION ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
