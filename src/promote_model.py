
import json
from datetime import datetime, timezone
from pathlib import Path

from mlflow.tracking import MlflowClient

MODEL_NAME = "debit-sungai-regressor"
ROOT = Path(__file__).resolve().parent.parent


def main():
    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    candidates = []

    for version in versions:
        if not version.run_id:
            continue

        run = client.get_run(version.run_id)
        val_rmse = run.data.metrics.get("val_rmse")

        if val_rmse is None:
            continue

        candidates.append({
            "version": int(version.version),
            "run_id": version.run_id,
            "val_rmse": float(val_rmse),
            "parameters": dict(run.data.params),
            "dataset": run.data.tags.get("data_file"),
            "data_tag": run.data.tags.get("data_tag"),
            "dvc_hash": run.data.tags.get(
                "dvc_processed_md5"
            ),
        })

    if len(candidates) < 2:
        raise RuntimeError(
            "Minimal dua versi model diperlukan."
        )

    for item in sorted(
        candidates, key=lambda x: x["version"]
    ):
        print(
            f"v{item['version']} | "
            f"validation RMSE: {item['val_rmse']:.4f}"
        )

    champion = min(
        candidates,
        key=lambda x: x["val_rmse"]
    )

    # Mencatat lifecycle model menggunakan tags.
    for item in candidates:
        client.set_model_version_tag(
            MODEL_NAME,
            str(item["version"]),
            "deployment_status",
            "candidate",
        )

    client.set_model_version_tag(
        MODEL_NAME,
        str(champion["version"]),
        "deployment_status",
        "production",
    )

    # Alias champion selalu mengarah ke versi terbaik.
    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        str(champion["version"]),
    )

    metadata = {
        "model_name": MODEL_NAME,
        "champion_version": champion["version"],
        "run_id": champion["run_id"],
        "validation_rmse": champion["val_rmse"],
        "parameters": champion["parameters"],
        "dataset": champion["dataset"],
        "data_tag": champion["data_tag"],
        "dvc_hash": champion["dvc_hash"],
        "selection_criterion": "lowest_validation_rmse",
        "promoted_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    output = ROOT / "docs" / "lk07_model_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nChampion: v{champion['version']}"
    )
    print("Metadata:", output)


if __name__ == "__main__":
    main()
