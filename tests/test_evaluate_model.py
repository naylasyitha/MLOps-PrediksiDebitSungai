
import math

import pytest

from src.evaluate_model import validate_model


@pytest.fixture
def evaluation_config():
    return {
        "model_name": "debit-sungai-regressor",
        "champion_version": "2",
        "metric_name": "val_rmse",
        "max_val_rmse": 14.0008310682955,
        "dataset_file": "dataset_20261101_151050.csv",
        "dvc_processed_md5": "e87c4864cd62451d1269902b73620419.dir",
    }


@pytest.fixture
def valid_tags():
    return {
        "data_file": "dataset_20261101_151050.csv",
        "dvc_processed_md5": "e87c4864cd62451d1269902b73620419.dir",
    }


def test_evaluation_passed_lower_rmse(evaluation_config, valid_tags):
    metrics = {"val_rmse": 13.5}

    result = validate_model(metrics, valid_tags, evaluation_config)

    assert result["status"] == "PASSED"
    assert result["eligible_for_registration"] is True


def test_evaluation_passed_equal_rmse(evaluation_config, valid_tags):
    metrics = {"val_rmse": 14.0008310682955}

    result = validate_model(metrics, valid_tags, evaluation_config)

    assert result["status"] == "PASSED"
    assert result["eligible_for_registration"] is True


def test_evaluation_failed_higher_rmse(evaluation_config, valid_tags):
    metrics = {"val_rmse": 15.0}

    result = validate_model(metrics, valid_tags, evaluation_config)

    assert result["status"] == "FAILED"
    assert result["eligible_for_registration"] is False


def test_evaluation_missing_rmse(evaluation_config, valid_tags):
    metrics = {}

    with pytest.raises(ValueError, match="tidak ditemukan"):
        validate_model(metrics, valid_tags, evaluation_config)


def test_evaluation_nan_rmse(evaluation_config, valid_tags):
    metrics = {"val_rmse": math.nan}

    with pytest.raises(ValueError, match="tidak valid"):
        validate_model(metrics, valid_tags, evaluation_config)


def test_evaluation_negative_rmse(evaluation_config, valid_tags):
    metrics = {"val_rmse": -5.0}

    with pytest.raises(ValueError, match="tidak valid"):
        validate_model(metrics, valid_tags, evaluation_config)


def test_evaluation_different_dataset(evaluation_config, valid_tags):
    metrics = {"val_rmse": 13.0}

    invalid_tags = valid_tags.copy()
    invalid_tags["data_file"] = "dataset_lain.csv"

    with pytest.raises(ValueError, match="Dataset"):
        validate_model(metrics, invalid_tags, evaluation_config)


def test_evaluation_different_dvc_hash(evaluation_config, valid_tags):
    metrics = {"val_rmse": 13.0}

    invalid_tags = valid_tags.copy()
    invalid_tags["dvc_processed_md5"] = "hash_berbeda.dir"

    with pytest.raises(ValueError, match="DVC"):
        validate_model(metrics, invalid_tags, evaluation_config)
