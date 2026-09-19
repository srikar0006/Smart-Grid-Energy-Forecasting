"""Prediction and generation-balance logic shared by the GUI and tests."""

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


MODEL_PATH = Path(__file__).parent / "artifacts_daily" / "daily_xgboost_model.json"
def validate_inputs(date_text, current_generation):
    try:
        date = dt.date.fromisoformat(date_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter the date in YYYY-MM-DD format.") from exc
    if not math.isfinite(current_generation) or current_generation < 0:
        raise ValueError("Current energy generation must be a non-negative number.")
    return date


def build_features(date_text, current_generation):
    date = validate_inputs(date_text, current_generation)
    timestamp = pd.Timestamp(date)
    day_of_year = timestamp.dayofyear
    values = {
        "current_energy_generation": float(current_generation),
        "date_ordinal": timestamp.toordinal(),
        "year": timestamp.year,
        "month": timestamp.month,
        "day_of_month": timestamp.day,
        "day_of_week": timestamp.dayofweek,
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
    }
    return pd.DataFrame([values])


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: .venv/bin/python train_daily_xgboost.py"
        )
    model = XGBRegressor()
    model.load_model(path)
    return model


def predict_realized_load(date_text, current_generation, model):
    features = build_features(date_text, current_generation)
    return float(model.predict(features)[0])


def calculate_balance(current_generation, predicted_load):
    if not math.isfinite(predicted_load) or predicted_load <= 0:
        raise ValueError("Predicted realized load must be greater than zero.")

    difference = float(current_generation - predicted_load)
    signed_percentage = difference / predicted_load * 100.0
    if abs(signed_percentage) < 0.01:
        status = "Balanced"
    elif difference > 0:
        status = "Overproducing"
    else:
        status = "Underproducing"
    return {
        "status": status,
        "difference": difference,
        "percentage": signed_percentage,
    }
