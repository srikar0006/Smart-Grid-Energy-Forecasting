"""Clean daily energy data and train the realized-load XGBoost model."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataframe_daily_cleaned.csv"
TARGET = "realized_load"
ARTIFACT_DIR = BASE_DIR / "artifacts_daily"
RANDOM_SEED = 42
INPUT_FEATURE = "current_energy_generation"


def clean_data(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path)
    report: dict[str, int] = {"input_rows": len(raw)}
    required = {"date", INPUT_FEATURE, TARGET}
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    # Ignore unrelated columns if a user adds them to the daily CSV later.
    df = df[["date", INPUT_FEATURE, TARGET]].copy()
    numeric_columns = [INPUT_FEATURE, TARGET]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    report["invalid_dates"] = int(df["date"].isna().sum())
    report["duplicate_dates"] = int(df["date"].duplicated(keep="last").sum())
    report["negative_measurement_rows"] = int((df[numeric_columns] < 0).any(axis=1).sum())
    report["rows_with_missing_values"] = int(df.isna().any(axis=1).sum())

    df.loc[(df[numeric_columns] < 0).any(axis=1), numeric_columns] = np.nan
    df = (
        df.dropna(subset=["date", *numeric_columns])
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )
    report["output_rows"] = len(df)
    report["rows_removed"] = report["input_rows"] - report["output_rows"]
    return df, report


def create_date_features(daily: pd.DataFrame) -> pd.DataFrame:
    dates = daily["date"]
    features = pd.DataFrame(
        {"current_energy_generation": daily["current_energy_generation"]}
    )
    features["date_ordinal"] = dates.map(pd.Timestamp.toordinal)
    features["year"] = dates.dt.year
    features["month"] = dates.dt.month
    features["day_of_month"] = dates.dt.day
    features["day_of_week"] = dates.dt.dayofweek
    features["day_of_year_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    features["day_of_year_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
    return features


def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    daily, cleaning_report = clean_data(DATA_PATH)
    # Keep the input file normalized after validation and cleaning.
    daily.to_csv(DATA_PATH, index=False)

    X = create_date_features(daily)
    y = daily[TARGET]
    test_start = int(len(daily) * 0.80)
    validation_start = int(test_start * 0.90)

    X_fit, y_fit = X.iloc[:validation_start], y.iloc[:validation_start]
    X_validation = X.iloc[validation_start:test_start]
    y_validation = y.iloc[validation_start:test_start]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

    model = XGBRegressor(
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.90,
        min_child_weight=3,
        reg_lambda=2,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=75,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    prediction = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, prediction)),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
        "best_iteration": int(model.best_iteration),
        "daily_rows": len(daily),
        "fit_rows": len(X_fit),
        "validation_rows": len(X_validation),
        "test_rows": len(X_test),
        "test_period": [
            daily["date"].iloc[test_start].isoformat(),
            daily["date"].iloc[-1].isoformat(),
        ],
        "features": list(X.columns),
        "training_data": DATA_PATH.name,
        "split": "chronological 72% fit / 8% validation / 20% test",
    }
    predictions = pd.DataFrame(
        {
            "date": daily["date"].iloc[test_start:].to_numpy(),
            "actual_realized_load": y_test.to_numpy(),
            "predicted_realized_load": prediction,
            "residual": y_test.to_numpy() - prediction,
        }
    )
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    model.save_model(ARTIFACT_DIR / "daily_xgboost_model.json")
    with (ARTIFACT_DIR / "daily_xgboost_model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)
    predictions.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
    importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (ARTIFACT_DIR / "cleaning_report.json").write_text(
        json.dumps(cleaning_report, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
