# Daily Energy Production Monitor

This project predicts daily average `realized_load` from a date and the daily
average `current_energy_generation`, using XGBoost. The desktop GUI also compares
generation with predicted load and reports whether the system is overproducing or
underproducing.

## GUI

Launch the application from this directory:

```bash
.venv/bin/python gui.py
```

Enter:

- A valid date in `YYYY-MM-DD` format
- Current energy generation as a non-negative number, in the same units as the CSV

The GUI displays predicted realized load, the signed unit difference, the signed
percentage difference, and the production status.

Use **Train model** inside the GUI to validate and clean
`dataframe_daily_cleaned.csv`, retrain XGBoost, refresh the evaluation artifacts,
and load the new model. Training runs in the background so the window remains
responsive.

```text
difference = current_energy_generation - predicted_realized_load
percentage = difference / predicted_realized_load * 100
```

- Positive: overproducing
- Negative: underproducing
- Within 0.01%: balanced

## Model

`current_energy_generation` is the sum of the 11 selected generation sources,
averaged across each day's 24 hourly observations. `realized_load` is averaged by
day as well. Biomass is excluded because it was not in the requested source list.

The final 20% of days are held out chronologically. The resulting test R² is
**0.717790**. See [DAILY_REPORT.md](DAILY_REPORT.md) for the complete evaluation.

## Data source

The hourly electricity load and generation data comes from the [German Electrical Load 2015–2024 dataset on Kaggle](https://www.kaggle.com/datasets/vsevolodnedora/german-electrical-load-2015-2024).

Retrain the model directly from `dataframe_daily_cleaned.csv` with:

```bash
.venv/bin/python train_daily_xgboost.py
```

Run tests with:

```bash
.venv/bin/python -m pytest -q
```

## Project structure

```text
Project3/
├── dataframe.csv                 # original hourly source data
├── dataframe_daily_cleaned.csv   # date, generation and realized-load daily means
├── artifacts_daily/              # trained model, metrics and evaluation outputs
├── gui.py                        # desktop interface
├── predict.py                    # prediction, validation and balance calculation
├── train_daily_xgboost.py        # cleaning, aggregation, training and evaluation
├── test_predict.py               # prediction and calculation tests
└── DAILY_REPORT.md               # model report
```
