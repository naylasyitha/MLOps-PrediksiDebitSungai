import pandas as pd
import pytest

from src.preprocess import (
    PreprocessingError,
    add_features,
    keep_latest_records,
    merge_sources,
    remove_invalid_values,
    save_dataset,
)


def test_remove_invalid_values():
    data = pd.DataFrame({
        "river_discharge": [25.0, -10.0],
        "precipitation": [5.0, -5.0],
        "soil_moisture": [0.5, 1.5],
    })

    result = remove_invalid_values(data.copy())

    assert result.loc[0, "river_discharge"] == 25.0
    assert result.loc[0, "precipitation"] == 5.0
    assert result.loc[0, "soil_moisture"] == 0.5
    assert pd.isna(result.loc[1, "river_discharge"])
    assert pd.isna(result.loc[1, "precipitation"])
    assert pd.isna(result.loc[1, "soil_moisture"])


def test_keep_latest_records():
    data = pd.DataFrame({
        "time": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "ingested_at": [
            "2026-01-01T08:00:00",
            "2026-01-01T10:00:00",
            "2026-01-02T08:00:00",
        ],
        "river_discharge": [10.0, 20.0, 30.0],
    })

    result = keep_latest_records(data)

    assert len(result) == 2
    assert result.loc[
        result["time"] == "2026-01-01", "river_discharge"
    ].iloc[0] == 20.0
    assert "ingested_at" not in result.columns


def test_merge_sources_without_matching_dates():
    flood = pd.DataFrame(
        {"river_discharge": [10.0]},
        index=pd.to_datetime(["2026-01-01"]),
    )
    weather = pd.DataFrame(
        {"precipitation": [5.0], "soil_moisture": [0.5]},
        index=pd.to_datetime(["2026-01-02"]),
    )

    with pytest.raises(PreprocessingError):
        merge_sources(flood, weather)


def test_add_features():
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    data = pd.DataFrame({
        "river_discharge": [
            10.0, 11.0, 12.0, 13.0,
            14.0, 15.0, 16.0, 17.0
        ],
        "precipitation": [1.0] * 8,
        "soil_moisture": [0.5] * 8,
    }, index=dates)

    result = add_features(data.copy())

    assert result.loc[dates[3], "river_discharge_lag1"] == 12.0
    assert result.loc[dates[3], "river_discharge_lag3"] == 10.0
    assert result.loc[dates[2], "precipitation_roll3"] == 3.0
    assert result.loc[dates[6], "precipitation_roll7"] == 7.0
    assert result.loc[
        dates[2], "soil_moisture_roll3"
    ] == pytest.approx(0.5)
    assert result.loc[dates[0], "is_rainy_season"] == 1


def test_save_dataset_prevents_overwrite(tmp_path):
    data = pd.DataFrame({"river_discharge": [10.0, 20.0]})
    output = save_dataset(data, tmp_path, "20260101_120000")

    assert output.exists()
    assert output.name == "dataset_20260101_120000.csv"

    with pytest.raises(PreprocessingError):
        save_dataset(data, tmp_path, "20260101_120000")
  