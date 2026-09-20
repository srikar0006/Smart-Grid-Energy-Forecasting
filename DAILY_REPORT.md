# Daily XGBoost energy-load report

## Requested transformation

Each output row represents one UTC calendar day. `current_energy_generation` is
calculated for every hour by summing the 11 requested source columns and is then
averaged across the 24 hours in that day. `realized_load` is also averaged across
the same 24 hours. All 3,549 days contain exactly 24 hourly observations.

`biomass` is excluded because it was not included in the latest requested list.

## Model result

- Features: generation, weekday, annual cycle, and realized-load lags of 1 and 7 days
- Split: chronological 72% fit, 8% early-stopping validation, 20% test
- Test period: 2022-10-10 through 2024-09-18
- Test days: 710
- **Test R²: 0.859138**
- MAE: 438.231
- RMSE: 577.558
- Out-of-range fallback R²: 0.752514

The application looks up lag values automatically; the user still enters only a
date and generation. If either lag is unavailable, prediction automatically uses
the fallback model.
