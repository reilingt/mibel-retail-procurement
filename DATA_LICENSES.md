# Data sources, licensing and the public dataset

This repository ships a dataset (`data/merged_2018_2024_hourly_final.csv`) so the pipeline
runs out of the box. Most columns are the real hourly series. Two columns (natural gas and
carbon prices) come from sources whose terms do not allow redistribution, so they are
**replaced with synthetic series** here. Everything else is real and reused under the terms
below. This file documents each source, its licence, how each column is defined, and what was
changed.

## Attribution

If you reuse this data, keep these notices:

- **Electricity prices**: Source [OMIE](https://www.omie.es). Public market data, reused with
  attribution per the OMIE legal notice.
- **Load, generation, flows and TSO forecasts**: Source [ENTSO-E Transparency Platform](https://transparency.entsoe.eu).
  Reused with attribution per the Transparency Platform Terms & Conditions.
- **Weather**: Generated using Copernicus Climate Change Service information (ERA5, 2018-2024),
  licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Neither the European
  Commission nor ECMWF is responsible for any use of this information.

## Columns

One row per hour, 2018-01-01 to 2024-12-31. `datetime` is local Lisbon time: the spring
daylight-saving hour does not exist and the autumn one appears twice (the pipeline keeps the
first). Portugal is the bidding zone throughout.

| Column(s) | Definition | Source | Licence / terms | In this repo |
|---|---|---|---|---|
| `da_price`, `da_price_es` | Day-ahead marginal price, Portugal / Spain, EUR/MWh | OMIE | Free reuse with source citation | **Real** |
| `id_price`, `id_price_es` | Intraday auction price, mean over the sessions covering the hour, EUR/MWh | OMIE | Free reuse with source citation | **Real** |
| `demand_mw` | Actual total load (consumption, excluding pumping and exports), MW | ENTSO-E Actual Total Load | Reuse with attribution | **Real** |
| `gen_wind_mw`, `gen_solar_mw`, `gen_gas_mw`, `gen_coal_mw`, `gen_biomass_mw`, `gen_hydro_ror_mw`, `gen_hydro_res_mw`, `gen_hydro_pump_mw` | Actual generation per type, MW (wind = onshore + offshore) | ENTSO-E Actual Generation per Type | Reuse with attribution | **Real** |
| `net_flow_mw` | Physical flow Portugal → Spain minus Spain → Portugal, MW (positive = export) | ENTSO-E Physical Flows | Reuse with attribution | **Real** |
| `ntc_pt_es_mw`, `ntc_es_pt_mw` | Day-ahead net transfer capacity in each direction, MW | ENTSO-E | Reuse with attribution | **Real** |
| `congestion_ratio` | abs(`net_flow_mw`) / NTC in the flow direction, capped at 2. Physical flows can exceed the commercial NTC, so values above 1 occur | derived | as its inputs | **Real** |
| `wind_forecast_mw`, `solar_forecast_mw` | TSO day-ahead forecast, MW. Hours without a day-ahead value take the forecast for the same hour one day earlier | ENTSO-E Generation Forecast | Reuse with attribution | **Real** |
| `load_fc_tso`, `load_fc_es` | TSO day-ahead total load forecast, Portugal / Spain, MW, published on D-1 and known at gate closure | ENTSO-E Day-ahead Total Load Forecast | Reuse with attribution | **Real** |
| `era5_wind_norte_l1`, `era5_t2m_*_l1` | 100 m wind speed (Minho) and 2 m temperature (Porto, Lisbon, Algarve), lagged 1 h | Copernicus / ECMWF ERA5 | CC-BY 4.0 | **Real** |
| `era5_tp_norte_l336`, `era5_tp_norte_l504` | Precipitation over the Douro/Minho basins, lagged 14 and 21 days | Copernicus / ECMWF ERA5 | CC-BY 4.0 | **Real** |
| `ttf_eur_mwh` | TTF gas front-month, previous trading day's close held for the whole day, EUR/MWh | originally Yahoo Finance / ICE Endex | Redistribution **not permitted** | **Synthetic** |
| `carbon_eur_t` | EUA carbon price (auction, else secondary market), previous day's value held for the whole day, EUR/t | originally ICAP / EEX | Redistribution **restricted** | **Synthetic** |

Notes on the definitions:

1. Nothing in a row is later than the hour it describes. Gas and carbon take the previous day's
   price, the wind and solar forecasts only use day-ahead values, and the pipeline adds a further
   24 h lag to every input (see the README).
2. Weather is a proxy. ERA5 is a reanalysis (observed weather) and stands in for the forecasts a
   retailer would have. The temperature series run about one to two hours early in summer relative
   to local time; solar radiation was dropped for the same reason, since the ENTSO-E solar forecast
   already carries that information.
3. Long forecast outages (up to 52 days in 2018 and early 2019) keep the TSO's intraday values.
   Both years lie outside every training window.

## The two synthetic columns

`ttf_eur_mwh` (gas) and `carbon_eur_t` (carbon) are **not real market data** in this repository.
Yahoo Finance's terms forbid redistributing data obtained through it, and the underlying ICE/EEX
and ICAP data carry their own redistribution restrictions. Rather than ship values we may not
publish, `data/make_public_dataset.py` replaces both columns with synthetic daily series that match the
real series' price level, volatility and persistence (a mean-reverting process in log space), so
the pipeline behaves realistically, but the numbers are fabricated and must not be read as market
prices.

What this means for the results: the tables in `results/` and the numbers in the README were
computed with the real gas and carbon series. Only aggregated metrics are published,
never the underlying series. Running the pipeline on the shipped dataset reproduces the same
conclusions and every procurement cost to within 0.12 EUR/MWh; the oracle, the naive baseline and
the ARX and profile forecast accuracy come out identical, because they do not use these two
columns. The gradient-boosting models use lagged gas and carbon as features, so they shift a
little on the synthetic series: the best HGB day-ahead price MAE is 15.3 instead of 15.0, and the
best strategy 64.97 instead of 64.86. The ranking, the significance and every conclusion are
unchanged.

To match the published figures to the cent, obtain the real TTF and EUA series yourself
(respecting their terms) and overwrite the two columns, using the previous-day rule above, before
running.

## Regenerating the public dataset

```bash
python data/make_public_dataset.py --in <real_merged.csv> --out data/merged_2018_2024_hourly_final.csv --seed 42
```

Deterministic given `--seed`. Only `ttf_eur_mwh` and `carbon_eur_t` are touched; every other column
is copied through unchanged.

SHA-256 of the shipped file, to check you have the same version:

```
16baa397c3441cc82965ec3ef6b8b781b63ad1df9bdd2e5f446e34164b01bdad  data/merged_2018_2024_hourly_final.csv
```

The lagged ERA5 columns are empty for their first hours by construction (for example
`era5_tp_norte_l504` has no value for the first 504 hours of 2018). This only affects 2018, which
is outside every training window.

*This is a good-faith reading of each provider's published terms as of 2026, not legal advice.*
