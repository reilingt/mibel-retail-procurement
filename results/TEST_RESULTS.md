# Verification log

Last checked: 2026-11-08. Python 3.13, default config, `SEED = 42`.

| Check | Result |
|---|---|
| Full pipeline (steps 1–8) on the original data, which produced every table in this folder | pass, exit 0, no exceptions |
| Full pipeline (steps 1–8) on the shipped public dataset | pass, exit 0, no exceptions |
| Re-run of steps 6–8 reproduces the result tables | byte-identical |
| `pytest` (17 tests: look-ahead, dataset, DST, train window, out-of-sample SHAP, settlement, statistics) | 17 passed |
| Look-ahead test against the previous code version | fails as intended, listing every lag-1 feature |
| Dataset tests against the previous dataset version | fail as intended (load incl. exports, intraday gas/carbon changes, radiation column) |
| Every number in the README compared with the tables in this folder | all match |
| `data/make_public_dataset.py --seed 42` regenerates the shipped dataset | identical SHA-256 |
| Result tables contain gas or carbon columns | no |
| Annual DA price means vs published OMIE averages (Spain 2018–2024) | match to the cent |

## Headline numbers (2024, EUR/MWh, surplus earns nothing)

```
oracle (perfect foresight)                 62.26
naive: profile load, all day-ahead         66.10
best: HGB load x fixed hourly DA/ID rule   64.86
saving vs naive                             1.27   95 % CI 0.91 to 1.71
ARX routing vs fixed hourly rule            0.003  95 % CI -0.11 to 0.14   (no difference)
best HGB vs ARX price routing               0.19   95 % CI  0.01 to 0.41   (p = 0.04)
price MAE: best HGB 14.98, ARX 17.29                (Diebold-Mariano p < 0.001)
load MAE: HGB 119 MW, free TSO forecast 191 MW, ARX 221 MW, profile 379 MW
battery: -3.9 in operation, +1.0 incl. CAPEX (300 EUR/kWh, 12 years)
```
