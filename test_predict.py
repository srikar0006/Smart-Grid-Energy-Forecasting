import datetime as dt
import math

import pytest

from predict import (
    build_features,
    calculate_balance,
    fetch_temperature,
    load_model,
    predict_realized_load,
)


def test_feature_values():
    features = build_features("2024-01-15", 14000, 5.5)
    assert features.loc[0, "current_energy_generation"] == 14000
    assert features.loc[0, "temperature_celsius"] == 5.5
    assert features.loc[0, "day_of_week"] == 0


@pytest.mark.parametrize("date", ["", "15-01-2024", "2024-02-30"])
def test_invalid_date(date):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        build_features(date, 14000, 5.5)


@pytest.mark.parametrize("generation", [-1, math.inf, math.nan])
def test_invalid_generation(generation):
    with pytest.raises(ValueError, match="non-negative"):
        build_features("2024-01-15", generation, 5.5)


@pytest.mark.parametrize("temperature", [math.inf, math.nan])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="Temperature"):
        build_features("2024-01-15", 14000, temperature)


def test_saved_temperature_lookup():
    assert math.isfinite(fetch_temperature("2024-01-15"))


def test_temperature_lookup_rejects_date_beyond_forecast_window():
    date = dt.date.today() + dt.timedelta(days=17)
    with pytest.raises(ValueError, match="16 days ahead"):
        fetch_temperature(date.isoformat())


def test_balance_status_and_percentage():
    over = calculate_balance(120, 100)
    under = calculate_balance(80, 100)
    assert over == {"status": "Overproducing", "difference": 20.0, "percentage": 20.0}
    assert under == {"status": "Underproducing", "difference": -20.0, "percentage": -20.0}


@pytest.mark.parametrize("generation", [95, 100, 105])
def test_balance_within_five_percent(generation):
    assert calculate_balance(generation, 100)["status"] == "Balanced"


def test_saved_model_predicts_positive_load():
    prediction = predict_realized_load(
        "2024-01-15",
        14000,
        5.5,
        load_model(),
        load_lag_1=15000,
        load_lag_7=14500,
    )
    assert prediction > 0


def test_fallback_model_predicts_without_lags():
    prediction = predict_realized_load(
        "2030-01-15", 14000, 5.5, load_model(), use_lags=False
    )
    assert prediction > 0


def test_lag_model_requires_manual_load_values():
    with pytest.raises(ValueError, match="yesterday"):
        predict_realized_load("2030-01-15", 14000, 5.5, load_model(), use_lags=True)


@pytest.mark.parametrize("lag", [-1, math.inf, math.nan])
def test_lag_model_rejects_invalid_load_values(lag):
    with pytest.raises(ValueError, match="non-negative"):
        predict_realized_load(
            "2024-01-15",
            14000,
            5.5,
            load_model(),
            load_lag_1=lag,
            load_lag_7=14500,
        )
