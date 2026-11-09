"""Step 4: hourly load forecasts for the test year (Profile, ARX, LEAR, DNN, HGB, HGB-ext, LSTM).
Writes load_fc.csv and load_accuracy.csv."""
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

import forecasting
from config import SEED

log = logging.getLogger(__name__)

BASE_COLUMNS = ["datetime", "date", "hour", "weekday", "is_holiday", "load"]
# ARX feature set and LEAR alphas come from config (cfg["arx_features_load"], cfg["lear_*"]).
# Fallback if no SHAP sets are passed in
NON_FEATURES = {"datetime", "date", "year", "dayofyear", "holiday_name",
                "load", "dayahead_price", "intraday_price",
                "wind", "solar", "flow_net_MW", "ttf", "eua",
                "gen_gas", "gen_coal", "hydro_ror", "congestion"}

# LSTM/DNN building blocks and feature-group constants are in forecasting.py (shared with step 5)


# Setup and helpers

def setup_tensorflow(seed=None):
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    log.info("TensorFlow %s, %d GPU(s). Models are placed on the CPU.", tf.__version__, len(gpus))


def is_done(load_fc, col):
    """A column counts as done if more than half of it is filled (used to resume from cache)."""
    return col in load_fc.columns and load_fc[col].notna().mean() > 0.5


def save_checkpoint(load_fc, path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        load_fc.to_csv(path, index=False)
        log.info("checkpoint saved: %d columns", len(load_fc.columns))
    except Exception as e:
        log.warning("checkpoint failed: %s", e)


def select_load_features(df, shap_sets_load):
    features = [c for c in (shap_sets_load or {}).get("all", []) if c in df.columns]
    if not features:
        log.warning("no SHAP sets for load, using all non-target columns")
        features = [c for c in df.columns if c not in NON_FEATURES]
    return features


def metrics(y_true, y_pred, name):
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"model": name, "mae_mw": np.nan, "rmse_mw": np.nan, "mape_pct": np.nan}
    mape = np.mean(np.abs(yp - yt) / np.where(yt == 0, np.nan, yt)) * 100
    return {
        "model": name,
        "mae_mw": round(mean_absolute_error(yt, yp), 1),
        "rmse_mw": round(mean_squared_error(yt, yp) ** 0.5, 1),
        "mape_pct": round(mape, 2),
    }


# Profile and ARX

def profile_forecast(df, load_fc, test_year):
    """Mean load per (weekday, hour, holiday) over all years before the test year."""
    history = df[df["year"] < test_year]
    profile = history.groupby(["weekday", "hour", "is_holiday"])["load"].mean().to_dict()
    fallback = history["load"].mean()
    preds = load_fc.apply(
        lambda r: profile.get((int(r["weekday"]), int(r["hour"]), bool(r["is_holiday"])), fallback), axis=1)
    return preds, len(history)


def arx_forecast(df, df_test, features, test_year):
    """24 OLS models, one per hour, trained on all years before the test year."""
    preds = pd.Series(np.nan, index=df_test.index)
    history = df[df["year"] < test_year]
    n_samples = 0
    if len(features) >= 2:
        for h in range(24):
            train_h = history[history["hour"] == h].dropna(subset=features + ["load"])
            test_h = df_test[df_test["hour"] == h].dropna(subset=features)
            if len(train_h) > 50 and len(test_h) > 0:
                model = LinearRegression().fit(train_h[features], train_h["load"])
                preds.loc[test_h.index] = model.predict(test_h[features])
                n_samples += len(train_h)
    return preds.values, n_samples


# LSTM


def lstm_forecast(features, mode, df, df_test, history, cfg, lstm):
    retrain_step = int(cfg["retrain_step"])
    retrain_days = int(cfg["lstm_retrain_days"])
    days = forecasting.mode_window_days(mode)
    seq = lstm["seq"]
    n_test = len(df_test)

    te_X_raw = df_test[features].ffill().bfill().values
    te_y_raw = df_test["load"].ffill().bfill().values.reshape(-1, 1)

    br = forecasting.lstm_branches(features, "load")
    preds = []
    n_retrains = 0
    model = None
    t0 = time.perf_counter()

    if mode.startswith("static"):
        t_end = pd.Timestamp(f"{cfg['test_year']}-01-01")
        train = df[(df["datetime"] >= t_end - pd.Timedelta(days=days)) & (df["datetime"] < t_end)]
        tr_X_raw = train[features].ffill().bfill().values
        tr_y_raw = train["load"].ffill().bfill().values.reshape(-1, 1)
        model, _, _, _ = forecasting.fit_lstm(tr_X_raw, tr_y_raw, br, lstm, "load_out", "multi_input_lstm_load")
        n_retrains = 1

        sc_X = MinMaxScaler().fit(tr_X_raw)
        sc_y = MinMaxScaler().fit(tr_y_raw)
        # context for each chunk is the seq hours right before it (training tail for the
        # first chunk, observed test hours afterwards), not a fixed last training week
        hist_X = np.vstack([tr_X_raw, te_X_raw])
        hist_y = np.vstack([tr_y_raw, te_y_raw])
        n_tr = len(tr_X_raw)
        for step in range(0, n_test, retrain_step):
            end = min(step + retrain_step, n_test)
            g = n_tr + step
            preds.extend(forecasting.predict_lstm(
                model, br, sc_y,
                sc_X.transform(hist_X[g - seq:g]), sc_y.transform(hist_y[g - seq:g]).ravel(),
                sc_X.transform(te_X_raw[step:end]), sc_y.transform(te_y_raw[step:end]).ravel(),
                seq))
    else:
        sc_X = sc_y = None
        last_retrain = None
        for step in tqdm(range(0, n_test, retrain_step), desc=f"  LSTM {mode}", ncols=80):
            end = min(step + retrain_step, n_test)
            now = df_test.iloc[step]["datetime"]

            if model is None or last_retrain is None or (now - last_retrain).days >= retrain_days:
                window = history[(history["datetime"] >= now - pd.Timedelta(days=days)) & (history["datetime"] < now)]
                X_win = window[features].ffill().bfill().values
                y_win = window["load"].ffill().bfill().values.reshape(-1, 1)
                if len(X_win) >= seq + 24:
                    model, sc_X, sc_y, _ = forecasting.fit_lstm(X_win, y_win, br, lstm,
                                                                "load_out", "multi_input_lstm_load")
                    last_retrain = now
                    n_retrains += 1

            if model is None:
                preds.extend([np.nan] * (end - step))
                continue

            ctx = history[history["datetime"] < now].tail(seq)
            preds.extend(forecasting.predict_lstm(
                model, br, sc_y,
                sc_X.transform(ctx[features].ffill().bfill().values),
                sc_y.transform(ctx[["load"]].ffill().bfill().values).ravel(),
                sc_X.transform(te_X_raw[step:end]), sc_y.transform(te_y_raw[step:end]).ravel(),
                seq))

    n_params = int(model.count_params()) if model is not None else 0
    branch_info = ", ".join(f"{n}={len(br[f'{n}_idx'])}" for n in br["branches"])
    meta = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "n_samples_train": days * 24,
        "n_features": len(features),
        "n_retrains": n_retrains,
        "epochs_or_iter": (f"{'rolling' if mode.startswith('rolling') else 'static'} · window={days}d · "
                           f"{n_retrains} retrain(s) · seq={seq}h · {n_params:,} params · branches: {branch_info}"),
    }
    return np.array(preds[:n_test]), meta


# LEAR

def lear_forecast(df_test, history, features, window_days, lear):
    """Lasso, Ridge and ElasticNet on asinh(load), retrained every day on a rolling window."""
    preds = {k: pd.Series(np.nan, index=df_test.index) for k in ["lasso", "ridge", "enet"]}
    runtime = {"lasso": 0.0, "ridge": 0.0, "enet": 0.0}
    iters = {"lasso": [], "enet": []}
    n_retrains = 0
    n_samples_last = 0

    for d in tqdm(sorted(df_test["date"].unique()), desc="  LEAR load", ncols=80):
        ts = pd.Timestamp(d)
        window = history[(history["datetime"] < ts) & (history["datetime"] >= ts - pd.Timedelta(days=window_days))
                         ].dropna(subset=features + ["load"])
        day = df_test[df_test["date"] == d].dropna(subset=features)
        if len(window) <= 500 or len(day) == 0:
            continue

        y = np.arcsinh(window["load"].values)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(window[features].values)
        X_te = scaler.transform(day[features].values)

        models = {
            "lasso": Lasso(alpha=lear["lasso_alpha"], max_iter=lear["max_iter"], random_state=SEED),
            "ridge": Ridge(alpha=lear["ridge_alpha"], random_state=SEED),
            "enet": ElasticNet(alpha=lear["enet_alpha"], l1_ratio=lear["enet_l1"],
                               max_iter=lear["max_iter"], random_state=SEED),
        }
        for name, model in models.items():
            t = time.perf_counter()
            model.fit(X_tr, y)
            preds[name].loc[day.index] = np.sinh(model.predict(X_te))
            if name in iters:
                iters[name].append(getattr(model, "n_iter_", 0))
            runtime[name] += time.perf_counter() - t

        n_retrains += 1
        n_samples_last = len(window)

    def meta(name, detail):
        return {"runtime_s": round(runtime[name], 2), "n_samples_train": n_samples_last,
                "n_features": len(features), "n_retrains": n_retrains, "epochs_or_iter": detail}

    avg = {k: (np.mean(v) if v else 0) for k, v in iters.items()}
    metas = {
        "LEAR_Lasso": meta("lasso", f"~{avg['lasso']:.0f} iter/retrain avg (max 20000)"),
        "LEAR_Ridge": meta("ridge", "closed-form (Cholesky)"),
        "LEAR_ElasticNet": meta("enet", f"~{avg['enet']:.0f} iter/retrain avg (max 20000)"),
    }
    return preds, metas


# DNN

def dnn_forecast(mode, features, df_test, history):
    """MLP 256-128, retrained every 7 days on a rolling window of the train-mode length."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    feats = [c for c in features if c in df_test.columns]
    preds = pd.Series(np.nan, index=df_test.index)
    t0 = time.perf_counter()
    n_retrains = 0
    n_samples_last = 0
    epochs = []
    n_params = 0
    days = forecasting.mode_window_days(mode)

    if len(feats) < 5:
        return preds, {"runtime_s": 0, "n_samples_train": 0, "n_features": len(feats),
                       "n_retrains": 0, "epochs_or_iter": "skipped (< 5 features)"}

    tf.random.set_seed(SEED)
    model = forecasting.build_dnn(len(feats))
    init_weights = model.get_weights()

    scaler = None
    last_date = None
    for d in tqdm(sorted(df_test["date"].unique()), desc=f" DNN {mode}", ncols=80):
        ts = pd.Timestamp(d)
        if last_date is None or (ts - pd.Timestamp(last_date)).days >= 7:
            window = history[(history["datetime"] < ts) & (history["datetime"] >= ts - pd.Timedelta(days=days))
                             ].dropna(subset=feats + ["load"])
            if len(window) > 500:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(window[feats])
                tf.random.set_seed(SEED)
                model.set_weights(init_weights)
                with forecasting.device_ctx():
                    hist = model.fit(X_tr, window["load"].values, epochs=500, batch_size=64,
                                     validation_split=0.1, verbose=0,
                                     callbacks=[EarlyStopping(patience=10, restore_best_weights=True, verbose=0)])
                last_date = d
                n_retrains += 1
                n_samples_last = len(window)
                epochs.append(len(hist.history.get("loss", [])))
                n_params = int(model.count_params())

        day = df_test[df_test["date"] == d].dropna(subset=feats)
        if scaler is not None and len(day) > 0:
            with forecasting.device_ctx():
                preds.loc[day.index] = model.predict(scaler.transform(day[feats]), verbose=0).ravel()

    meta = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "n_samples_train": n_samples_last,
        "n_features": len(feats),
        "n_retrains": n_retrains,
        "epochs_or_iter": (f"rolling · window={days}d · {n_retrains} retrain(s) · "
                           f"avg {int(np.mean(epochs)) if epochs else 0} epochs · {n_params:,} params"),
    }
    return preds, meta


# HGB

def rolling_hgb(data, df_test, features, window_days, hgb_params, retrain_every=7):
    """Walk-forward HGB: retrain every 7 days on a rolling window, predict each test day."""
    preds = np.full(len(df_test), np.nan)
    model = None
    last_date = None
    meta = {"n_retrains": 0, "n_samples_last": 0, "n_iter_last": 0, "train_time_s": 0.0, "infer_time_s": 0.0}

    for d in tqdm(sorted(df_test["date"].unique()), desc="  Rolling HGB load", ncols=80):
        if model is None or (last_date and (pd.Timestamp(d) - pd.Timestamp(last_date)).days >= retrain_every):
            cutoff = pd.Timestamp(d)
            mask = (data["datetime"] >= cutoff - pd.Timedelta(days=window_days)) & (data["datetime"] < cutoff)
            train = data[mask].dropna(subset=["load"])  # HGB handles NaN features itself
            if len(train) > 200:
                t = time.perf_counter()
                model = HistGradientBoostingRegressor(**hgb_params).fit(train[features], train["load"])
                meta["train_time_s"] += time.perf_counter() - t
                last_date = d
                meta["n_retrains"] += 1
                meta["n_samples_last"] = len(train)
                meta["n_iter_last"] = getattr(model, "n_iter_", hgb_params["max_iter"])

        day_mask = df_test["date"] == d
        if model is not None:
            t = time.perf_counter()
            preds[day_mask.values] = model.predict(df_test.loc[day_mask, features])
            meta["infer_time_s"] += time.perf_counter() - t
    return preds, meta


def hgb_forecast(mode, features, df, df_train, df_test, profile, hgb_params, test_year):
    """HGB for one train mode. Gaps are filled with the profile forecast."""
    t0 = time.perf_counter()
    days = forecasting.mode_window_days(mode)

    if mode.startswith("rolling"):
        data = pd.concat([df_train, df_test]).sort_values("datetime")
        preds, m = rolling_hgb(data, df_test, features, days, hgb_params)
        return pd.Series(preds, index=df_test.index).fillna(profile), {
            "runtime_s": round(time.perf_counter() - t0, 2),
            "train_time_s": round(m["train_time_s"], 2),
            "infer_time_s": round(m["infer_time_s"], 2),
            "n_samples_train": m["n_samples_last"],
            "n_features": len(features),
            "n_retrains": m["n_retrains"],
            "epochs_or_iter": f"{m['n_iter_last']} (last retrain, window={days}d)",
        }

    years = list(range(test_year - days // 365, test_year))
    train = df[df["year"].isin(years)].dropna(subset=["load"])
    if len(train) == 0:
        raise ValueError(f"hgb_forecast({mode!r}): no training rows for years {years}")
    t = time.perf_counter()
    model = HistGradientBoostingRegressor(**hgb_params).fit(train[features], train["load"])
    train_s = time.perf_counter() - t
    t = time.perf_counter()
    preds = pd.Series(model.predict(df_test[features]), index=df_test.index).fillna(profile)
    infer_s = time.perf_counter() - t
    return preds, {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "train_time_s": round(train_s, 2),
        "infer_time_s": round(infer_s, 2),
        "n_samples_train": len(train),
        "n_features": len(features),
        "n_retrains": 1,
        "epochs_or_iter": f"{getattr(model, 'n_iter_', hgb_params['max_iter'])} iter (window={days}d)",
    }


def add_extended_lags(df, df_train, df_test, history):
    """Add load_lag25..load_lag167 so HGB sees the same week of load history as the LSTM.
    The lags are added to df, df_train, df_test and history so rolling HGB-ext trains on
    them too (df_train precedes df_test chronologically, so it is the first len(df_train) rows)."""
    full = pd.concat([df_train, df_test]).sort_values("datetime").reset_index(drop=True)
    new_cols = [f"load_lag{lag}" for lag in range(25, 168) if f"load_lag{lag}" not in full.columns]
    if not new_cols:
        return df, df_train, df_test, history

    lags = pd.DataFrame({c: full["load"].shift(int(c.replace("load_lag", ""))) for c in new_cols}, index=full.index)
    full = pd.concat([full, lags], axis=1)

    n_train, n_test = len(df_train), len(df_test)
    df_train = df_train.assign(**{c: lags[c].values[:n_train] for c in new_cols}).copy()
    df_test = df_test.assign(**{c: lags[c].values[-n_test:] for c in new_cols}).copy()
    df_test["date"] = pd.to_datetime(df_test["datetime"]).dt.date
    history = history.copy().assign(**{c: lags[c].values for c in new_cols}).copy()
    df = df.merge(full[["datetime"] + new_cols], on="datetime", how="left")
    log.info("HGB-ext: added %d lag columns", len(new_cols))
    return df, df_train, df_test, history


# Main

def run(cfg, df, df_train, df_test, shap_sets_load):
    # Keras and scikit-learn print many UserWarnings in the grids, mute them only here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        families = set(cfg.get("model_families", ["lear", "dnn", "hgb", "hgb_ext", "lstm"]))
        if "lstm" in families or "dnn" in families:
            setup_tensorflow(cfg["seed"])  # only needed for the TensorFlow-based families
            forecasting.set_tf_device(cfg.get("tf_device", "/CPU:0"))
        return forecast_all(cfg, df, df_train, df_test, shap_sets_load)


def forecast_all(cfg, df, df_train, df_test, shap_sets_load):

    # work on a clean RangeIndex so load_fc, the resume cache and every per-column
    # assignment stay positionally aligned (avoids the index-reset corruption)
    df_test = df_test.reset_index(drop=True)

    modes = cfg["train_mode"]
    modes = modes if isinstance(modes, list) else [modes]
    primary = modes[0]
    test_year = cfg["test_year"]
    out_dir = Path(cfg["out_dir"])
    cache_path = out_dir / "load_fc.csv"
    hgb_params = dict(
        max_iter=int(cfg["hgb_max_iter"]),
        max_depth=int(cfg["hgb_max_depth"]),
        learning_rate=float(cfg["hgb_learning_rate"]),
        min_samples_leaf=int(cfg["hgb_min_samples_leaf"]),
        random_state=int(cfg["hgb_random_state"]),
    )
    lstm = {
        "seq": int(cfg["lstm_seq_len_load"]),
        "units": int(cfg["lstm_units"]),
        "epochs": int(cfg["lstm_epochs"]),
        "batch_size": int(cfg["lstm_batch_size"]),
    }
    meta = {}

    load_features = select_load_features(df, shap_sets_load)
    arx_features = [c for c in cfg["arx_features_load"] if c in df.columns]
    lear_features = [c for c in arx_features + ["hour"] if c in df.columns]
    lear_params = {"lasso_alpha": cfg["lear_lasso_alpha"], "ridge_alpha": cfg["lear_ridge_alpha"],
                   "enet_alpha": cfg["lear_enet_alpha"], "enet_l1": cfg["lear_enet_l1_ratio"],
                   "max_iter": int(cfg["lear_max_iter"])}
    shap_sets = {k: [c for c in v if c in df_test.columns] for k, v in (shap_sets_load or {}).items()}
    if not shap_sets:
        shap_sets = {"all": [c for c in load_features if c in df_test.columns]}
    log.info("load forecast: modes %s, %d features, SHAP sets %s", modes, len(load_features), list(shap_sets))

    history = pd.concat([df_train, df_test]).sort_values("datetime").reset_index(drop=True)

    load_fc = df_test[BASE_COLUMNS].copy()
    load_fc["load_actual"] = load_fc["load"]
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["datetime"])
        new_cols = [c for c in cached.columns if c not in load_fc.columns]
        if new_cols:
            # merge resets the index; later assignments by the df_test index would give NaN
            load_fc = load_fc.merge(cached[["datetime"] + new_cols], on="datetime", how="left")
            log.info("resumed %d columns from %s", len(new_cols), cache_path)

    if not is_done(load_fc, "fc_profile"):
        t0 = time.perf_counter()
        load_fc["fc_profile"], n_hist = profile_forecast(df, load_fc, test_year)
        meta["profile"] = {"runtime_s": round(time.perf_counter() - t0, 2), "n_samples_train": n_hist,
                           "n_features": 3, "n_retrains": 1,
                           "epochs_or_iter": "N/A (empirical mean, all years before test_year)"}

    if not is_done(load_fc, "fc_arx"):
        t0 = time.perf_counter()
        load_fc["fc_arx"], n_samples = arx_forecast(df, df_test, arx_features, test_year)
        meta["ARX"] = {"runtime_s": round(time.perf_counter() - t0, 2), "n_samples_train": n_samples,
                       "n_features": len(arx_features), "n_retrains": 24,
                       "epochs_or_iter": "closed-form OLS (24 hourly LR, all years before test_year)"}

    # TSO baseline: the free ENTSO-E day-ahead load forecast, calibrated to our load level
    # with a single additive bias from the training years (what a retailer gets for nothing)
    if "load_fc_tso" in df_test.columns and not is_done(load_fc, "fc_tso"):
        t0 = time.perf_counter()
        hist = df[df["year"] < test_year].dropna(subset=["load", "load_fc_tso"])
        bias = float((hist["load"] - hist["load_fc_tso"]).mean()) if len(hist) else 0.0
        load_fc["fc_tso"] = pd.Series(df_test["load_fc_tso"].values + bias,
                                      index=load_fc.index).fillna(load_fc["fc_profile"])
        meta["TSO"] = {"runtime_s": round(time.perf_counter() - t0, 2), "n_samples_train": len(hist),
                       "n_features": 1, "n_retrains": 0,
                       "epochs_or_iter": f"ENTSO-E day-ahead load forecast, additive bias {bias:+.0f} MW"}

    families = set(cfg.get("model_families", ["lear", "dnn", "hgb", "hgb_ext", "lstm"]))

    # LSTM (one failure stops the whole LSTM grid, as in the thesis runs)
    if "lstm" in families:
        try:
            for mode in modes:
                for label, feats in shap_sets.items():
                    feats = [c for c in feats if c in df_test.columns]
                    col = f"fc_lstm_{mode}_shap_{label}"
                    if is_done(load_fc, col):
                        continue
                    log.info("LSTM %s / %s (%d features)", mode, label, len(feats))
                    load_fc[col], meta[f"lstm_{mode}_shap_{label}"] = lstm_forecast(
                        feats, mode, df, df_test, history, cfg, lstm)
        except Exception as e:
            log.warning("LSTM skipped: %s: %s", type(e).__name__, e)
        save_checkpoint(load_fc, cache_path)
    else:
        log.info("LSTM grid skipped (not in cfg['model_families'])")

    if "lear" in families:
        if not all(is_done(load_fc, c) for c in ["fc_lear", "fc_lear_ridge", "fc_lear_enet"]):
            # window length comes from the first train mode only
            preds, lear_meta = lear_forecast(df_test, history, lear_features,
                                             forecasting.mode_window_days(primary), lear_params)
            load_fc["fc_lear"] = preds["lasso"].values
            load_fc["fc_lear_ridge"] = preds["ridge"].values
            load_fc["fc_lear_enet"] = preds["enet"].values
            meta.update(lear_meta)
            save_checkpoint(load_fc, cache_path)
    else:
        log.info("LEAR skipped (not in cfg['model_families'])")

    if "dnn" in families:
        for mode in modes:
            for label, feats in shap_sets.items():
                feats = [c for c in feats if c in df_test.columns]
                col = f"fc_dnn_{mode}_shap_{label}"
                if is_done(load_fc, col):
                    continue
                log.info("DNN %s / %s (%d features)", mode, label, len(feats))
                load_fc[col], meta[f"dnn_{mode}_shap_{label}"] = dnn_forecast(mode, feats, df_test, history)
        save_checkpoint(load_fc, cache_path)
    else:
        log.info("DNN grid skipped (not in cfg['model_families'])")

    if "hgb" in families:
        for mode in modes:
            for label, feats in shap_sets.items():
                feats = [c for c in feats if c in df_test.columns]
                col = f"fc_hgb_{mode}_shap_{label}"
                if is_done(load_fc, col):
                    continue
                log.info("HGB %s / %s (%d features)", mode, label, len(feats))
                load_fc[col], meta[f"hgb_{mode}_shap_{label}"] = hgb_forecast(
                    mode, feats, df, df_train, df_test, load_fc["fc_profile"], hgb_params, test_year)
        save_checkpoint(load_fc, cache_path)
    else:
        log.info("HGB grid skipped (not in cfg['model_families'])")

    if "hgb_ext" in families:
        df, df_train, df_test, history = add_extended_lags(df, df_train, df_test, history)
        ext_lags = [f"load_lag{lag}" for lag in range(25, 168) if f"load_lag{lag}" in df_test.columns]
        shap_sets_ext = {k: [c for c in v if c in df_test.columns] + [c for c in ext_lags if c not in v]
                         for k, v in shap_sets.items()}
        for mode in modes:
            for label, feats in shap_sets_ext.items():
                feats = [c for c in feats if c in df_test.columns]
                col = f"fc_hgb_ext_{mode}_shap_{label}"
                if is_done(load_fc, col):
                    continue
                log.info("HGB-ext %s / %s (%d features)", mode, label, len(feats))
                try:
                    preds, m = hgb_forecast(mode, feats, df, df_train, df_test, load_fc["fc_profile"],
                                            hgb_params, test_year)
                except Exception:
                    log.exception("HGB-ext failed: %s", col)
                    continue
                load_fc[col] = preds.values
                meta[f"hgb_ext_{mode}_shap_{label}"] = m
                save_checkpoint(load_fc, cache_path)
        save_checkpoint(load_fc, cache_path)
    else:
        log.info("HGB-ext grid skipped (not in cfg['model_families'])")

    y_true = load_fc["load_actual"].values
    rows = [metrics(y_true, load_fc["fc_profile"].values, "profile")]
    for col, name in [("fc_arx", "ARX"), ("fc_tso", "TSO"), ("fc_lear", "LEAR_Lasso"), ("fc_lear_ridge", "LEAR_Ridge"),
                      ("fc_lear_enet", "LEAR_ElasticNet"), ("fc_dnn", "DNN_MLP"),
                      ("fc_hgb", "hgb_primary"), ("fc_lstm", "lstm_primary")]:
        if col in load_fc.columns:
            rows.append(metrics(y_true, load_fc[col].values, name))
    for col in load_fc.columns:
        if col.startswith(("fc_hgb_", "fc_lstm_", "fc_dnn_")) and col not in ("fc_hgb", "fc_lstm", "fc_dnn"):
            rows.append(metrics(y_true, load_fc[col].values, col.replace("fc_", "")))
    load_accuracy = pd.DataFrame(rows).sort_values("mae_mw")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        load_fc.to_csv(out_dir / "load_fc.csv", index=False)
        load_accuracy.to_csv(out_dir / "load_accuracy.csv", index=False)
        log.info("saved load_fc.csv (%d rows, %d columns) to %s", len(load_fc), len(load_fc.columns), out_dir)
    except Exception as e:
        log.warning("saving failed: %s", e)

    log.info("accuracy (MW):\n%s", load_accuracy.to_string(index=False))
    return {"load_fc": load_fc, "load_accuracy": load_accuracy, "load_training_meta": meta}
