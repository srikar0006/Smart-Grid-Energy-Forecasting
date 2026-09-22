"""Predict daily load and compare it with generation."""

import datetime as dt
import json
import math
import pickle
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts_daily" / "daily_xgboost_model.pkl"
WEATHER_PATH = ROOT / "weather_data.csv"
LATITUDE = 50.1109
LONGITUDE = 8.6821


def parse_date(date_text):
    try:
        return dt.date.fromisoformat(date_text)
    except (TypeError, ValueError) as error:
        raise ValueError("Enter the date in YYYY-MM-DD format.") from error


def validate_inputs(date_text, generation, temperature):
    date = parse_date(date_text)
    if not math.isfinite(generation) or generation < 0:
        raise ValueError("Current energy generation must be a non-negative number.")
    if not math.isfinite(temperature):
        raise ValueError("Temperature must be a valid number.")
    return date


def build_features(date_text, generation, temperature):
    """Build the features available for every date."""
    date = validate_inputs(date_text, generation, temperature)
    timestamp = pd.Timestamp(date)
    return pd.DataFrame(
        [
            {
                "current_energy_generation": float(generation),
                "temperature_celsius": float(temperature),
                "day_of_week": timestamp.dayofweek,
                "day_of_year_sin": np.sin(2 * np.pi * timestamp.dayofyear / 365.25),
                "day_of_year_cos": np.cos(2 * np.pi * timestamp.dayofyear / 365.25),
            }
        ]
    )


def fetch_temperature(date_text):
    """Return local historical temperature or fetch it from Open-Meteo."""
    date = parse_date(date_text)
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["date"])
    weather["date"] = pd.to_datetime(weather["date"], utc=True).dt.date
    saved = weather.set_index("date")["temperature_celsius"].get(date)
    if not pd.isna(saved):
        return float(saved)

    if date > dt.date.today() + dt.timedelta(days=16):
        raise ValueError(
            "Temperature forecasts are available only up to 16 days ahead. "
            "Enter the expected temperature manually for this date."
        )

    historical = date < dt.date.today() - dt.timedelta(days=5)
    endpoint = (
        "https://archive-api.open-meteo.com/v1/archive"
        if historical
        else "https://api.open-meteo.com/v1/forecast"
    )
    query = urlencode(
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": date.isoformat(),
            "end_date": date.isoformat(),
            "daily": "temperature_2m_mean",
            "timezone": "UTC",
        }
    )
    try:
        with urlopen(f"{endpoint}?{query}", timeout=15) as response:
            result = json.load(response)
        return float(result["daily"]["temperature_2m_mean"][0])
    except (
        HTTPError,
        URLError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(f"Temperature is unavailable for {date.isoformat()}.") from error


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Trained model not found. Run train_daily_xgboost.py")
    with path.open("rb") as file:
        return pickle.load(file)


def predict_realized_load(
    date_text,
    generation,
    temperature,
    models,
    use_lags=True,
    load_lag_1=None,
    load_lag_7=None,
):
    validate_inputs(date_text, generation, temperature)
    features = build_features(date_text, generation, temperature)
    if not use_lags:
        return float(models["fallback"].predict(features)[0])

    if (
        load_lag_1 is None
        or load_lag_7 is None
        or not math.isfinite(load_lag_1)
        or not math.isfinite(load_lag_7)
        or load_lag_1 < 0
        or load_lag_7 < 0
    ):
        raise ValueError(
            "Enter valid non-negative realized load values for yesterday "
            "and seven days ago."
        )
    features["load_lag_1"] = float(load_lag_1)
    features["load_lag_7"] = float(load_lag_7)
    return float(models["lag"].predict(features)[0])


def calculate_balance(generation, predicted_load):
    if not math.isfinite(predicted_load) or predicted_load <= 0:
        raise ValueError("Predicted realized load must be greater than zero.")
    difference = float(generation - predicted_load)
    percentage = difference / predicted_load * 100
    if abs(percentage) <= 5:
        status = "Balanced"
    elif difference > 0:
        status = "Overproducing"
    else:
        status = "Underproducing"
    return {"status": status, "difference": difference, "percentage": percentage}
