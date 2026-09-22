# Daily XGBoost energy-load report

## Requested transformation

Each output row represents one UTC calendar day. `current_energy_generation` is
calculated for every hour by summing the 11 requested source columns and is then
averaged across the 24 hours in that day. `realized_load` is also averaged across
the same 24 hours. All 3,549 days contain exactly 24 hourly observations.

`biomass` is excluded because it was not included in the latest requested list.

## Model result

- Features: generation, daily mean temperature, weekday, annual cycle, and
  realized-load lags of 1 and 7 days
- Split: chronological 72% fit, 8% early-stopping validation, 20% test
- Test period: 2022-10-10 through 2024-09-18
- Test days: 710
- **Test R²: 0.871173**
- MAE: 406.598
- RMSE: 552.333
- Out-of-range fallback R²: 0.775441

The user enters a date, generation, and either manually entered or automatically
fetched temperature, then selects the lag or fallback model. Lag mode reveals
required inputs for realized load one and seven days earlier. Fallback mode does
not require those values.
