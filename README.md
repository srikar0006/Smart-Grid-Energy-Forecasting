# Smart Grid Energy Forecasting

This program predicts the average electricity load for a day and compares that
prediction with the supplied generation. It then reports whether the system is
**overproducing**, **underproducing**, or **balanced**.

The desktop interface stays simple: the user enters only a date and the current
daily-average generation. Historical load values are looked up automatically.

## How the program works

```text
Date + current generation
           |
           v
Validate the two inputs
           |
           v
Build generation and calendar features
           |
           v
Look for realized load 1 and 7 days earlier
       /                     \
 both found              either missing
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

The choice between the two models is automatic and does not add any GUI inputs.

## Input data

Training reads `dataframe_daily_cleaned.csv` directly. It does not clean,
aggregate, or rewrite the file. The required columns are:

| Column | Meaning |
|---|---|
| `date` | UTC calendar date |
| `current_energy_generation` | Average generation for that day |
| `realized_load` | Average measured load for that day |

The file contains 3,549 daily rows. `current_energy_generation` was produced by
summing the 11 selected generation sources for each hourly observation and then
averaging the 24 observations in each day. Biomass is excluded from that sum.

The original hourly data is in `dataframe.csv`. The current training code does
not read that file.

## Model features

Both models use four compact features:

| Feature | Purpose |
|---|---|
| `current_energy_generation` | Generation entered by the user |
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
created with `shift(1)` and `shift(7)`. During prediction, the program searches
`dataframe_daily_cleaned.csv` for the exact earlier dates.

For example, a prediction for `2024-01-15` uses realized load from:

- `2024-01-14` as `load_lag_1`
- `2024-01-08` as `load_lag_7`

If either date is unavailable, the program uses the fallback model.

## Training process

Training is implemented in `train_daily_xgboost.py`:

1. Read the already-cleaned daily CSV and parse its dates.
2. Reserve the final 20% of days for testing.
3. Use the preceding 8% for early-stopping validation.
4. Use the first 72% for model fitting.
5. Train the fallback model from generation and calendar features.
6. Add the two historical load lags and train the main lag model.
7. Evaluate the lag model on the untouched final 20%.
8. Save both models and the evaluation results.

The split is chronological rather than random. The model is therefore evaluated
on later dates than those used for fitting, which is closer to real forecasting.

XGBoost builds regression trees sequentially. Each new tree attempts to correct
errors made by the earlier trees. Training permits up to 1,500 trees, but early
stopping ends it when validation RMSE has not improved for 75 rounds.

The saved result stopped at iteration 613.

## Current evaluation

The held-out test period is 2022-10-10 through 2024-09-18 and contains 710 days.

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Lag model | **0.859138** | 438.231 | 577.558 |
| Fallback model | 0.752514 | — | — |

R² measures how much of the variation in load is explained by the model. MAE is
the average absolute prediction error, while RMSE penalizes larger errors more
strongly. MAE and RMSE use the same units as `realized_load`.

## Prediction process

Prediction is implemented in `predict.py`:

1. `validate_inputs()` checks the ISO date and rejects negative, infinite, or
   missing generation values.
2. `build_features()` creates generation, weekday, and annual-cycle features.
3. `add_lags()` looks up the loads from one and seven days earlier.
4. `predict_realized_load()` selects the lag or fallback model.
5. `calculate_balance()` compares the prediction with generation.

The balance calculation is:

```text
difference = current_generation - predicted_realized_load
percentage = difference / predicted_realized_load * 100
```

The status rules are:

- Positive difference: **Overproducing**
- Negative difference: **Underproducing**
- Absolute difference below 0.01%: **Balanced**

The displayed difference and percentage are signed, so a leading `+` means
excess generation and a leading `-` means insufficient generation.

## Desktop interface

`gui.py` provides the Tkinter interface. On startup it loads the saved model
bundle and test R². Pressing **Predict realized load** calls the shared prediction
and balance functions and displays the result.

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

The tests cover feature creation, lag lookup, fallback behavior, input
validation, balance calculations, and loading the saved model.

## Project structure

```text
Smart Grid Energy Forecasting/
├── artifacts_daily/
│   ├── daily_xgboost_model.pkl
│   ├── metrics.json
│   └── test_predictions.csv
├── dataframe.csv
├── dataframe_daily_cleaned.csv
├── gui.py
├── predict.py
├── train_daily_xgboost.py
├── test_predict.py
├── requirements.txt
└── DAILY_REPORT.md
```

The hourly electricity load and generation data originated from the
[German Electrical Load 2015–2024 dataset on Kaggle](https://www.kaggle.com/datasets/vsevolodnedora/german-electrical-load-2015-2024).
