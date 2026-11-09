# Algorithmic procurement for an electricity retailer in MIBEL

[![tests](https://github.com/reilingt/mibel-retail-procurement/actions/workflows/tests.yml/badge.svg)](https://github.com/reilingt/mibel-retail-procurement/actions/workflows/tests.yml)

A portfolio project built from my MSc Business Analytics thesis (Católica Lisbon, 2026). I
rewrote the thesis work as a reproducible pipeline and kept only what held up under testing.
Code is in `src/`, result tables in `results/`.

## The problem

A retailer buys its customers' electricity every hour. For each hour it can buy in the
day-ahead auction or wait for intraday, and whatever it gets wrong settles at the imbalance
price. So there are two forecasting problems (how much customers use, which market is cheaper)
and one investment question (does a battery help?). I turn each choice into a cost: the all-in
procurement cost in EUR/MWh over the 2024 test year, models trained on 2022-2023, with 95%
intervals from a bootstrap.

## What I found

1. Forecasting beats buying blindly by 1.27 EUR/MWh (about 0.65 million EUR a year for a 1%
   retailer), well above noise.
2. But almost all of that saving is free. Buying each hour in the market that was cheaper on
   average, a fixed rule with no model, already captures 0.82 of it. A trained price forecast
   adds nothing on top.
3. The more accurate model is not the more useful one. HGB predicts prices better than linear
   ARX, yet ARX picks the cheaper market more often and buys cheaper. Routing only needs the
   sign of the price gap, not its size.
4. A better load forecast barely matters under these settlement rules. The free ENTSO-E
   grid-operator forecast is about as accurate as a tuned model and buys no cheaper than a crude
   weekday/hour average.
5. A battery loses money at 2024 prices: about 3.9 EUR/MWh saved while running, but about 1.0
   more once the battery itself is counted.

## Results

Test year 2024, retailer share 1% of Portuguese consumption (514 GWh), cost in EUR/MWh, lower
is better. Surplus energy is either worthless (zero) or sold at 70% / 90% of the intraday price.
A strategy has two parts: a load forecast (how much to buy) and a routing rule (day-ahead or
intraday).

Routing decides most of the saving. With the best load forecast fixed:

| DA/ID routing (best load forecast) | zero | 70% | 90% |
|---|--:|--:|--:|
| Oracle (perfect foresight, lower bound) | 62.26 | 62.26 | 62.26 |
| Fixed hourly rule (learned on history) | 64.86 | 64.53 | 64.44 |
| ARX price forecast | 64.87 | 64.54 | 64.45 |
| Best HGB price forecast | 65.07 | 64.74 | 64.65 |
| Everything in day-ahead | 65.74 | 65.41 | 65.32 |

The load forecast adds a little on top, barely more than its accuracy suggests. With the fixed
routing rule:

| Load forecast | load MAE (MW) | zero | 70% | 90% |
|---|--:|--:|--:|
| HGB (uses the free TSO forecast as a feature) | 119 | 64.86 | 64.53 | 64.44 |
| ENTSO-E day-ahead forecast (free, as-is) | 191 | 65.34 | 64.68 | 64.49 |
| Weekday/hour profile (naive) | 379 | 65.28 | 64.62 | 64.43 |

The numbers behind the five points, with 95% intervals from a block bootstrap over the 366 test
days:

1. The best strategy is 1.27 EUR/MWh cheaper than buying the naive profile load entirely
   day-ahead (interval 0.91 to 1.71, p < 0.001), about 0.65 million EUR a year at this size, and
   still 0.82 to 0.91 cheaper when surplus can be sold.
2. The hourly market pattern is worth 0.82 EUR/MWh on its own. A price forecast adds nothing:
   ARX matches the fixed rule (difference 0.003, interval -0.11 to 0.14) and agrees with it in
   82% of hours.
3. The free TSO load forecast has an MAE of 191 MW, between ARX (221) and HGB, and is the single
   most important feature for the HGB load model, which cuts the MAE to 119 MW (3.3% to 2.0%).
   It is twice as accurate as the naive profile yet costs the same to procure with (65.34 vs
   65.28).
4. Going from the naive profile to HGB (a third of the error) saves only 0.42 EUR/MWh at zero
   surplus and nothing once surplus can be sold. Shortfalls settle at the intraday price, on
   average only 0.09 EUR/MWh above day-ahead in 2024, so under-forecasting is almost free. Real
   imbalance prices carry penalties that are not modelled here.
5. HGB forecasts the DA price better than ARX (MAE 15.0 vs 17.3, Diebold-Mariano p < 0.001), yet
   picks the cheaper market less often (56.6% vs 59.1%) and routes 0.19 EUR/MWh more expensively
   (interval 0.01 to 0.41, p = 0.04). Retraining HGB daily on a 6-month window
   (`price_hgb_daily_retrain=True`) sharpens the forecast to 13.4 MAE but still routes worse, so
   the procurement result does not move.
6. The battery saves about 3.9 EUR/MWh in operation, but at 2024 CAPEX (300 EUR/kWh over 12
   years) the annualised investment leaves the all-in cost about 1.0 EUR/MWh above the
   no-battery case.

All tables are in [`results/`](results/), the significance tests in
[`results/significance.csv`](results/significance.csv).

## How the pipeline works

Each step is one file in `src/`, run in order by `run_all.py`.

1. `01_data_prep.py`: loads the hourly dataset (2018-2024) and builds the feature table.
   Calendar features, Portuguese holidays and bridge days, lagged and rolling load/price/
   fundamentals, Spanish price and spread lags, the ENTSO-E day-ahead load forecasts (PT and ES,
   published on D-1) and ERA5 weather. Every lag is at least 24h old, since a day-ahead bid for
   day D is placed at noon on D-1. Splits train/test and ranks features with SHAP on a held-out
   part of the training window.
2. `02_eda_load.py`: load analysis (monthly patterns, peak days and hours, holiday effects).
3. `03_eda_price.py`: price analysis (day-ahead vs intraday levels and spread, by hour and over
   the year).
4. `04_load_forecast.py`: hourly load forecasts with up to seven model families (Profile, ARX,
   LEAR, DNN, HGB, HGB-ext, LSTM), plus the free ENTSO-E TSO forecast as a baseline, each
   retrained walk-forward through the year.
5. `05_price_forecast.py`: the same families for the DA and ID price, plus the routing decision
   as a fixed hourly rule and a forecast-driven rule.
6. `06_procurement.py`: buys `share × forecast` of the load in the chosen market for every
   load/price pair, with shortfalls at the ID price and surplus at 0 / 70 / 90% of it. Reports
   every combination, the naive baselines and the perfect-foresight oracle.
7. `07_bess.py`: adds a battery with three rule-based strategies across six sizes, with
   efficiency, CAPEX annuity, O&M and grid fees.
8. `08_significance.py`: compares the best strategy against the alternatives with a moving-block
   bootstrap over days, and tests the price-accuracy gap with Diebold-Mariano.

## Repository layout

```
.
├── README.md
├── DATA_LICENSES.md          # sources, licences, the two synthetic columns, checksum
├── CITATION.cff
├── requirements.txt          # core pipeline
├── requirements/
│   ├── nn.txt                # + TensorFlow for the DNN/LSTM families
│   └── dev.txt               # + pytest
├── data/
│   ├── merged_2018_2024_hourly_final.csv  # hourly dataset (see DATA_LICENSES.md)
│   └── make_public_dataset.py  # builds the shipped dataset (synthesises gas/carbon)
├── results/                  # result tables the README is based on
├── tests/                    # look-ahead, dataset, settlement and statistics tests
└── src/
    ├── config.py             # every setting, including the single random seed
    ├── 01_data_prep.py … 08_significance.py
    ├── forecasting.py        # shared LSTM/DNN building blocks
    ├── outputs.py
    └── run_all.py            # runs the steps in order
```

## Running it

```bash
pip install -r requirements.txt
python src/run_all.py --data-dir data --out-dir outputs     # all steps, about 30 min on a laptop CPU
python src/run_all.py --steps 6 7 8 --out-dir outputs       # re-run the economics from saved forecasts
pip install -r requirements/dev.txt && pytest                # tests, a few seconds
```

By default the pipeline trains Profile, ARX, HGB and HGB-ext over the `static_2y` and
`rolling_12m` windows, all on the CPU, and the committed result tables use only this subset, so
they reproduce exactly. One `SEED` in `config.py` controls every random source.

The LSTM and DNN families are much heavier and practical only on a GPU; I trained them on a Colab
GPU runtime. To add them, enable a GPU, install `requirements/nn.txt`, set `tf_device="auto"` and
widen `model_families` and `train_mode` in `src/config.py`. GPU training is non-deterministic even
with the seed.

Reproducibility: the shipped dataset has synthetic gas and carbon prices (licensing, see
[DATA_LICENSES.md](DATA_LICENSES.md)); every other column is the real series. Running on it
gives the same conclusions and reproduces every procurement cost to within 0.12 EUR/MWh, with
the oracle, naive baseline and ARX/profile accuracy identical to the cent. Only the
gradient-boosting models shift slightly (best HGB price MAE 15.3 instead of 15.0, best strategy
64.97 instead of 64.86), since they use gas and carbon as features.

### In Google Colab

```python
!git clone https://github.com/reilingt/mibel-retail-procurement.git
%cd mibel-retail-procurement
!pip install holidays shap          # the rest ships with Colab
!python src/run_all.py --data-dir data --out-dir outputs
```

For the LSTM and DNN families, switch the Colab runtime to a GPU and
`pip install -r requirements/nn.txt` before running.

## Data

Electricity prices (OMIE), load, generation, flows, the day-ahead load forecasts (ENTSO-E) and
weather (Copernicus ERA5) are the real hourly series, reused under each provider's terms. The
gas and carbon columns come from sources that do not permit redistribution and are replaced with
synthetic series of matching level and dynamics. Provenance and licences are in
[DATA_LICENSES.md](DATA_LICENSES.md).

## Methodology notes

1. No look-ahead: every lagged feature is at least 24h old and rolling windows end at t-24, so a
   forecast uses only what is known at gate closure. A test perturbs the last 24h of every input
   and checks nothing moves.
2. Weather and renewables are proxies: ERA5 reanalysis and the D-1 ENTSO-E wind/solar forecasts
   stand in for the forecasts a retailer would have, and are slightly optimistic.
3. Out-of-sample feature selection: SHAP ranks features on a held-out tail of the training
   window and never sees the test year.
4. Settlement without penalties: shortfalls at the ID price, surplus at 0 / 70 / 90% of it. Real
   imbalance prices include penalties, so the value of load accuracy here is a lower bound.
5. The battery is a rule-based add-on, not an optimised dispatch, so it is a sensitivity check
   rather than a core claim.
6. One test year (2024): the bootstrap intervals cover day-to-day noise within the year, not how
   a different year would turn out.

## License

Code under the MIT License ([LICENSE](LICENSE)). The data keeps each provider's own terms,
documented per column in [DATA_LICENSES.md](DATA_LICENSES.md); the shipped gas and carbon columns
are synthetic, everything else is real.
