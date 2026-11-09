"""Settings for steps 1 to 8. Defaults run the fast subset; comments note the full grid."""
import os

# one seed for every random source
SEED = 42


def build_cfg(**overrides):
    cfg = {
        # data_dir holds merged_2018_2024_hourly_final.csv (+ optional datasets/ and cache/ for ERA5)
        "data_dir": os.environ.get("MIBEL_DATA_DIR", "data"),
        "out_dir": os.environ.get("MIBEL_OUT_DIR", "outputs"),

        "test_year": 2024,
        # train windows per model; first one defines df_train
        # full grid: ["static_2y", "static_1y", "rolling_6m", "rolling_12m"]
        "train_mode": ["static_2y", "rolling_12m"],
        "share": 0.01,  # retailer share of system load
        "surplus_mode": "zero",
        "imbalance_price_factor": 0.7,
        # None = evaluate every hour with a full forecast set, else "YYYY-MM-DD"
        "eval_start_date": None,
        "surplus_scenarios": [
            {"label": "zero", "surplus_mode": "zero", "imbalance_price_factor": 0.00},
            {"label": "sur70", "surplus_mode": "imbalance", "imbalance_price_factor": 0.70},
            {"label": "sur90", "surplus_mode": "imbalance", "imbalance_price_factor": 0.90},
        ],

        # one feature set per cumulative-importance threshold; full grid [0.85, 0.90, 0.95, 0.99, 1.00]
        "shap_thresholds": [0.90, 1.00],

        # families trained on top of Profile and ARX (always on)
        # full grid adds "lear", "dnn", "lstm" (several hours)
        "model_families": ["hgb", "hgb_ext"],

        # HGB (also the SHAP reference model)
        "hgb_max_iter": 300,
        "hgb_max_depth": 6,
        "hgb_learning_rate": 0.05,
        "hgb_min_samples_leaf": 30,
        "hgb_random_state": SEED,

        # ARX/LEAR features; LEAR adds "hour" and runs on arcsinh(target)
        # load ARX stays price-free, price ARX adds DA/ID and Spanish lags
        "arx_features_load": ["load_lag24", "load_lag168", "load_rmean24", "load_rstd24",
                              "weekday", "is_holiday"],
        "arx_features_price": ["dayahead_price_lag24", "dayahead_price_lag168", "intraday_price_lag24",
                               "load_lag24", "spread_lag24", "da_price_es_lag24", "da_price_es_lag168",
                               "id_price_es_lag24", "id_price_es_lag168", "weekday", "is_holiday"],
        "lear_lasso_alpha": 0.001,
        "lear_ridge_alpha": 1.0,
        "lear_enet_alpha": 0.001,
        "lear_enet_l1_ratio": 0.5,
        "lear_max_iter": 20000,

        # LSTM
        "lstm_seq_len_load": 168,
        "lstm_seq_len_price": 168,
        "lstm_units": 64,
        "lstm_epochs": 50,
        "lstm_batch_size": 64,
        "retrain_step": 168,  # hours per LSTM prediction chunk
        "lstm_retrain_days": 30,
        "seed": SEED,
        # "/CPU:0" keeps runs deterministic; "auto" lets TF use a GPU
        "tf_device": "/CPU:0",

        # BESS
        "bess_configs": [
            {"name": "2h/10MW", "duration_h": 2, "P_max_MW": 10},
            {"name": "2h/20MW", "duration_h": 2, "P_max_MW": 20},
            {"name": "2h/40MW", "duration_h": 2, "P_max_MW": 40},
            {"name": "4h/10MW", "duration_h": 4, "P_max_MW": 10},
            {"name": "4h/20MW", "duration_h": 4, "P_max_MW": 20},
            {"name": "4h/40MW", "duration_h": 4, "P_max_MW": 40},
        ],
        "bess_load_models": ["fc_hgb_ext_static_2y_shap_p90", "fc_arx"],
        "bess_price_model": "fc_arx_da",
        "bess_eta": 0.90,  # round-trip efficiency
        "bess_soc_min": 0.10,
        "bess_soc_max": 0.90,
        "bess_cycles_per_day": 2.0,
        # 300 EUR/kWh over 12 years ~ 2024 utility-scale Li-ion
        "bess_capex_eur_per_kwh": 300,
        "bess_lifetime_years": 12,
        "bess_discount_rate": 0.06,
        "bess_om_pct_capex": 0.01,
        "bess_grid_fee_eur_mwh": 1.0,
    }
    cfg.update(overrides)
    return cfg
