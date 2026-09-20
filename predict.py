"""Predict daily load and compare it with generation."""

import datetime as dt
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts_daily" / "daily_xgboost_model.pkl"
DATA_PATH = ROOT / "dataframe_daily_cleaned.csv"
LAGS = (1, 7)


def validate_inputs(date_text, generation):
    try:
        date = dt.date.fromisoformat(date_text)
    except (TypeError, ValueError) as error:
        raise ValueError("Enter the date in YYYY-MM-DD format.") from error
    if not math.isfinite(generation) or generation < 0:
        raise ValueError("Current energy generation must be a non-negative number.")
    return date


def build_features(date_text, generation):
    """Build the four features available for every date."""
    date = validate_inputs(date_text, generation)
    timestamp = pd.Timestamp(date)
    return pd.DataFrame(
        [
            {
                "current_energy_generation": float(generation),
                "day_of_week": timestamp.dayofweek,
                "day_of_year_sin": np.sin(2 * np.pi * timestamp.dayofyear / 365.25),
                "day_of_year_cos": np.cos(2 * np.pi * timestamp.dayofyear / 365.25),
            }
        ]
    )


def add_lags(features, date):
    """Add yesterday's and last week's loads when both are available."""
    history = pd.read_csv(DATA_PATH, usecols=["date", "realized_load"], parse_dates=["date"])
    history["date"] = pd.to_datetime(history["date"], utc=True).dt.date
    loads = history.set_index("date")["realized_load"]
    values = {lag: loads.get(date - dt.timedelta(days=lag)) for lag in LAGS}
    if any(pd.isna(value) for value in values.values()):
        return None
    for lag, value in values.items():
        features[f"load_lag_{lag}"] = float(value)
    return features


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Trained model not found. Run train_daily_xgboost.py")
    with path.open("rb") as file:
        return pickle.load(file)


def predict_realized_load(date_text, generation, models):
    date = validate_inputs(date_text, generation)
    features = build_features(date_text, generation)
    lagged = add_lags(features.copy(), date)
    model = models["lag"] if lagged is not None else models["fallback"]
    return float(model.predict(lagged if lagged is not None else features)[0])


def calculate_balance(generation, predicted_load):
    if not math.isfinite(predicted_load) or predicted_load <= 0:
        raise ValueError("Predicted realized load must be greater than zero.")
    difference = float(generation - predicted_load)
    percentage = difference / predicted_load * 100
    if abs(percentage) < 0.01:
        status = "Balanced"
    elif difference > 0:
        status = "Overproducing"
    else:
        status = "Underproducing"
    return {"status": status, "difference": difference, "percentage": percentage}
