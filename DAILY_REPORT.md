# Daily XGBoost energy-load report

## Requested transformation

Each output row represents one UTC calendar day. `current_energy_generation` is
calculated for every hour by summing the 11 requested source columns and is then
averaged across the 24 hours in that day. `realized_load` is also averaged across
the same 24 hours. All 3,549 days contain exactly 24 hourly observations.

`biomass` is excluded because it was not included in the latest requested list.

## Model result

- Features: date-derived values and `current_energy_generation` only
- Split: chronological 72% fit, 8% early-stopping validation, 20% test
- Test period: 2022-10-10 through 2024-09-18
- Test days: 710
- **Test R²: 0.717790**
- MAE: 661.477
- RMSE: 817.494

This is a stricter future-period score. The earlier hourly R² is not comparable
because that model also used prior realized-load measurements and separate
generation columns.
