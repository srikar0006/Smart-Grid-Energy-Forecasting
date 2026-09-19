import math

import pytest

from predict import build_features, calculate_balance, load_model, predict_realized_load


def test_feature_values():
    features = build_features("2024-01-15", 14000)
    assert features.loc[0, "current_energy_generation"] == 14000
    assert features.loc[0, "year"] == 2024
    assert features.loc[0, "month"] == 1
    assert features.loc[0, "day_of_month"] == 15


@pytest.mark.parametrize("date", ["", "15-01-2024", "2024-02-30"])
def test_invalid_date(date):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        build_features(date, 14000)


@pytest.mark.parametrize("generation", [-1, math.inf, math.nan])
def test_invalid_generation(generation):
    with pytest.raises(ValueError, match="non-negative"):
        build_features("2024-01-15", generation)


def test_balance_status_and_percentage():
    over = calculate_balance(120, 100)
    under = calculate_balance(80, 100)
    assert over == {"status": "Overproducing", "difference": 20.0, "percentage": 20.0}
    assert under == {"status": "Underproducing", "difference": -20.0, "percentage": -20.0}


def test_saved_model_predicts_positive_load():
    prediction = predict_realized_load("2024-01-15", 14000, load_model())
    assert prediction > 0
