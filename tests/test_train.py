from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.train import (
    PersistenceBaseline,
    apply_model_defaults,
    build_model,
    evaluate,
    find_latest_dataset,
    rmse,
    split_by_time,
    split_features_target,
)


def test_find_latest_dataset(tmp_path):
    (tmp_path / "dataset_20260101_120000.csv").write_text(
        "date,river_discharge\n2026-01-01,10\n"
    )
    latest = tmp_path / "dataset_20260102_120000.csv"
    latest.write_text("date,river_discharge\n2026-01-02,20\n")

    result = find_latest_dataset(tmp_path)

    assert result == latest


def test_find_latest_dataset_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_dataset(tmp_path)


def test_split_by_time():
    data = pd.DataFrame({"value": range(100)})

    train, validation, test = split_by_time(data)

    assert len(train) == 70
    assert len(validation) == 15
    assert len(test) == 15
    assert train["value"].iloc[-1] == 69
    assert validation["value"].iloc[0] == 70
    assert test["value"].iloc[0] == 85


def test_split_features_target():
    data = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "river_discharge": [10.0, 20.0],
        "precipitation": [1.0, 2.0],
    })

    features, target = split_features_target(data)

    assert list(features.columns) == ["precipitation"]
    assert target.tolist() == [10.0, 20.0]


def test_rmse():
    observed = np.array([10.0, 20.0, 30.0])
    predicted = np.array([10.0, 18.0, 32.0])

    result = rmse(observed, predicted)

    assert result == pytest.approx(np.sqrt(8.0 / 3.0))


def test_evaluate():
    observed = np.array([10.0, 20.0, 30.0])
    predicted = np.array([10.0, 18.0, 32.0])

    metrics = evaluate(
        observed, predicted, high_flow_threshold=25.0
    )

    assert metrics["rmse"] == pytest.approx(np.sqrt(8.0 / 3.0))
    assert metrics["mae"] == pytest.approx(4.0 / 3.0)
    assert metrics["nse"] == pytest.approx(0.96)
    assert metrics["rmse_high_flow"] == pytest.approx(2.0)


def test_build_random_forest_and_baseline():
    args = SimpleNamespace(
        model="rf",
        n_estimators=20,
        max_depth=4,
        min_samples_leaf=1,
        seed=42,
    )

    model = build_model(args)

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 20
    assert model.max_depth == 4

    args.model = "baseline"
    baseline = build_model(args)

    assert isinstance(baseline, PersistenceBaseline)

    features = pd.DataFrame({
        "river_discharge_lag1": [10.0, 20.0]
    })

    assert baseline.predict(features).tolist() == [10.0, 20.0]
