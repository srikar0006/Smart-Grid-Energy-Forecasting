"""Train compact daily-load models from the prepared CSV."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "combined.csv"
ARTIFACT_DIR = ROOT / "artifacts_daily"
LAGS = (1, 7)


def base_features(data):
    """Small feature set that works for any date."""
    date = data["date"]
    return pd.DataFrame(
        {
            "current_energy_generation": data["current_energy_generation"],
            "temperature_celsius": data["temperature_celsius"],
            "day_of_week": date.dt.dayofweek,
            "day_of_year_sin": np.sin(2 * np.pi * date.dt.dayofyear / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * date.dt.dayofyear / 365.25),
        }
    )


def make_model():
    return XGBRegressor(
        n_estimators=1500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.90,
        min_child_weight=5,
        reg_lambda=4,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=75,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def train(features, target, fit_end, test_start):
    """Fit with chronological validation and score the final 20%."""
    model = make_model()
    model.fit(
        features.iloc[:fit_end],
        target.iloc[:fit_end],
        eval_set=[(features.iloc[fit_end:test_start], target.iloc[fit_end:test_start])],
        verbose=False,
    )
    prediction = model.predict(features.iloc[test_start:])
    actual = target.iloc[test_start:]
    metrics = {
        "r2": float(r2_score(actual, prediction)),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "best_iteration": int(model.best_iteration),
    }
    return model, prediction, metrics


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    data["date"] = pd.to_datetime(data["date"], utc=True)
    target = data["realized_load"]
    test_start = int(len(data) * 0.80)
    fit_end = int(test_start * 0.90)

    fallback_model, _, fallback_metrics = train(
        base_features(data), target, fit_end, test_start
    )

    lagged = base_features(data)
    for lag in LAGS:
        lagged[f"load_lag_{lag}"] = target.shift(lag)
    # Only the first seven rows lack lag history; split dates stay unchanged.
    valid_start = max(LAGS)
    lag_model, prediction, metrics = train(
        lagged.iloc[valid_start:].reset_index(drop=True),
        target.iloc[valid_start:].reset_index(drop=True),
        fit_end - valid_start,
        test_start - valid_start,
    )

    with (ARTIFACT_DIR / "daily_xgboost_model.pkl").open("wb") as file:
        pickle.dump({"lag": lag_model, "fallback": fallback_model}, file)

    actual = target.iloc[test_start:]
    pd.DataFrame(
        {
            "date": data["date"].iloc[test_start:].to_numpy(),
            "actual_realized_load": actual.to_numpy(),
            "predicted_realized_load": prediction,
            "residual": actual.to_numpy() - prediction,
        }
    ).to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)

    metrics.update(
        {
            "daily_rows": len(data),
            "test_rows": len(actual),
            "test_period": [
                data["date"].iloc[test_start].isoformat(),
                data["date"].iloc[-1].isoformat(),
            ],
            "features": list(lagged.columns),
            "lags": list(LAGS),
            "fallback_r2": fallback_metrics["r2"],
        }
    )
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
