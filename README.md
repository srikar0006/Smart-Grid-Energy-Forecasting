# Smart Grid Energy Forecasting

This program predicts the average electricity load for a day and compares that
prediction with the supplied generation. It then reports whether the system is
**overproducing**, **underproducing**, or **balanced**.

The user enters a date, current daily-average generation, and temperature.
Temperature can be typed manually or fetched automatically. When the lag model
is selected, the user also enters the realized loads from one and seven days
before the prediction date.

## How the program works

```text
Date + current generation + temperature
           |
           v
Validate the inputs
           |
           v
Build generation, weather, and calendar features
           |
           v
Choose the prediction model in the GUI
       /                     \
 Lag selected          Fallback selected
     |                         |
     v                         v
Enter 1-day and            No previous-load
7-day load values            inputs
     |                         |
     v                         v
 Lag model                Fallback model
       \                     /
        predicted realized load
                    |
                    v
 Compare generation with predicted load
                    |
                    v
 Overproducing / Underproducing / Balanced
```

Two XGBoost models are stored together:

- The **lag model** is the main model. It uses the loads from one day and seven
  days before the requested date and has the higher test accuracy.
- The **fallback model** does not need load history. It allows prediction for
  dates outside the range of the local dataset.

The GUI lets the user choose which model to use.

## Input data

Training reads `combined.csv` directly. This file combines the cleaned energy
data with daily temperature:

| Column | Meaning |
|---|---|
| `date` | UTC calendar date |
| `current_energy_generation` | Average generation for that day |
| `realized_load` | Average measured load for that day |
| `temperature_celsius` | Mean air temperature for that day in °C |

The file contains 3,549 daily rows. `current_energy_generation` was produced by
summing the 11 selected generation sources for each hourly observation and then
averaging the 24 observations in each day. Biomass is excluded from that sum.

`weather_data.csv` contains the `date` and `temperature_celsius` columns.
`combined.csv` contains all three cleaned-energy columns plus temperature. Both
files have exactly 3,549 matching dates and no missing temperatures.

The original hourly energy data is in `dataframe.csv`. The current training code
does not read that file.

## Weather data source

Temperature comes from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).
The location is Frankfurt, Germany (`50.1109° N, 8.6821° E`), used as a
consistent central-Germany weather proxy. Open-Meteo's hourly `temperature_2m`
measurement represents air temperature two metres above ground.

Hourly values were requested in UTC for 2015-01-01 through 2024-09-18. The 24
hourly readings for each date were averaged to create `temperature_celsius` in
`weather_data.csv`. This was then joined by date with
`dataframe_daily_cleaned.csv` to create `combined.csv`.

For dates already present in `weather_data.csv`, the GUI reads the saved local
value and does not require internet access. For other historical dates it uses
Open-Meteo's archive endpoint. For recent or forecast dates it uses the
[Open-Meteo Forecast API](https://open-meteo.com/en/docs) and requests daily mean
temperature. Open-Meteo may not provide forecasts for dates too far into the
future; the GUI reports that the temperature is unavailable in that case.

## Model features

Both models use five compact features:

| Feature | Purpose |
|---|---|
| `current_energy_generation` | Generation entered by the user |
| `temperature_celsius` | Daily mean temperature entered or fetched by the user |
| `day_of_week` | Represents weekday and weekend demand patterns |
| `day_of_year_sin` | First half of a continuous annual cycle |
| `day_of_year_cos` | Second half of the annual cycle |

Sine and cosine encode annual seasonality without creating an artificial break
between December 31 and January 1.

The main lag model also uses:

| Feature | Meaning |
|---|---|
| `load_lag_1` | Actual realized load one day earlier |
| `load_lag_7` | Actual realized load seven days earlier |

These are historical values, not future information. During training they are
created with `shift(1)` and `shift(7)`. During prediction, the user supplies both
values in the GUI when the lag model is selected.

For example, a lag-model prediction for `2024-01-15` requires the user to enter:

- `2024-01-14` as `load_lag_1`
- `2024-01-08` as `load_lag_7`

Both fields must contain finite, non-negative values. If those values are not
known, the user can select the fallback model instead.

## Training process

Training is implemented in `train_daily_xgboost.py`:

1. Read `combined.csv` and parse its dates.
2. Reserve the final 20% of days for testing.
3. Use the preceding 8% for early-stopping validation.
4. Use the first 72% for model fitting.
5. Train the fallback model from generation, temperature, and calendar features.
6. Add the two historical load lags and train the main lag model.
7. Evaluate the lag model on the untouched final 20%.
8. Save both models and the evaluation results.

The split is chronological rather than random. The model is therefore evaluated
on later dates than those used for fitting, which is closer to real forecasting.

XGBoost builds regression trees sequentially. Each new tree attempts to correct
errors made by the earlier trees. Training permits up to 1,500 trees, but early
stopping ends it when validation RMSE has not improved for 75 rounds.

The saved result stopped at iteration 1,114.

## Current evaluation

The held-out test period is 2022-10-10 through 2024-09-18 and contains 710 days.

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Weather + lag model | **0.871173** | 406.598 | 552.333 |
| Weather fallback model | 0.775441 | — | — |

R² measures how much of the variation in load is explained by the model. MAE is
the average absolute prediction error, while RMSE penalizes larger errors more
strongly. MAE and RMSE use the same units as `realized_load`.

## Prediction process

Prediction is implemented in `predict.py`:

1. `validate_inputs()` checks the ISO date, generation, and temperature.
2. `fetch_temperature()` first checks `weather_data.csv`; if the date is not
   stored locally, it requests the value from Open-Meteo.
3. `build_features()` creates generation, temperature, weekday, and annual-cycle
   features.
4. `predict_realized_load()` uses the model selected in the GUI. Lag mode adds
   the two manually entered earlier loads; fallback mode requires neither.
5. `calculate_balance()` compares the prediction with generation.

The balance calculation is:

```text
difference = current_generation - predicted_realized_load
percentage = difference / predicted_realized_load * 100
```

The status rules are:

- Positive difference: **Overproducing**
- Negative difference: **Underproducing**
- Percentage difference from −5% through +5%: **Balanced**

The displayed difference and percentage are signed, so a leading `+` means
excess generation and a leading `-` means insufficient generation.

## Desktop interface

`gui.py` provides the Tkinter interface. On startup it loads the saved model
bundle and both test R² values. **Fetch temperature for date** fills the
temperature field from the local weather CSV or Open-Meteo. The value can also be
edited manually. **Predict consumption** calls the shared prediction
and balance functions and displays the result.

The input and training panel has its own vertical scrollbar. It can be scrolled
with the scrollbar or mouse wheel when the lag fields make the panel taller than
the window.

The **Prediction model** controls provide two choices:

- **Lag model** uses realized load from one and seven days earlier and normally
  provides higher accuracy. Selecting it reveals two required load input fields.
- **Fallback model** uses generation, temperature, and calendar features only.
  Selecting it hides the two lag fields. Choose it when the earlier loads are
  unknown.

Pressing **Train model** starts training in a background thread so the window
does not freeze. A thread-safe queue reports completion back to the Tkinter main
thread, which then reloads the new model and score.

Start the GUI with:

```bash
.venv/bin/python gui.py
```

Enter:

- A date in `YYYY-MM-DD` format
- A non-negative daily-average generation value in the dataset's units
- A finite daily-mean temperature in °C, entered manually or fetched
- For lag mode, non-negative realized load values from one and seven days earlier

## Saved artifacts

Training writes only three files to `artifacts_daily/`:

| File | Contents |
|---|---|
| `daily_xgboost_model.pkl` | Lag and fallback XGBoost models |
| `metrics.json` | Model score, feature list, lags, and test period |
| `test_predictions.csv` | Date, actual load, prediction, and residual for each test day |

## Commands

Install dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Train both models:

```bash
.venv/bin/python train_daily_xgboost.py
```

Run all tests:

```bash
.venv/bin/python -m pytest -q
```

The tests cover weather lookup, feature creation, manual lag validation, fallback
behavior, balance calculations, and loading the saved model.

## Project structure

```text
Smart Grid Energy Forecasting/
├── artifacts_daily/
│   ├── daily_xgboost_model.pkl
│   ├── metrics.json
│   └── test_predictions.csv
├── dataframe.csv
├── dataframe_daily_cleaned.csv
├── weather_data.csv
├── combined.csv
├── gui.py
├── predict.py
├── train_daily_xgboost.py
├── test_predict.py
├── requirements.txt
└── DAILY_REPORT.md
```

The hourly electricity load and generation data originated from the
[German Electrical Load 2015–2024 dataset on Kaggle](https://www.kaggle.com/datasets/vsevolodnedora/german-electrical-load-2015-2024).
