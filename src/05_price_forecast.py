"""Step 5: DA and ID price forecasts for the test year, plus the DA/ID routing decision.
Writes price_fc.csv, price_accuracy.csv and id_accuracy.csv."""
import logging
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

BASE_COLUMNS = ["datetime", "date", "hour", "weekday", "is_holiday", "dayahead_price", "intraday_price"]
# ARX feature set and LEAR alphas come from config (cfg["arx_features_price"], cfg["lear_*"]).
NON_FEATURES = {"datetime", "date", "year", "dayofyear", "holiday_name",
                "load", "dayahead_price", "intraday_price",
                "wind", "solar", "flow_net_MW", "ttf", "eua",
                "gen_gas", "gen_coal", "hydro_ror", "congestion"}
ROLLING_DAYS = {"rolling_12m": 365, "rolling_6m": 182, "rolling_3m": 91}
STATIC_YEARS = {"static_2y": 2, "static_1y": 1}

# LSTM/DNN building blocks and feature-group constants are in forecasting.py (shared with step 4)


# Helpers


def accuracy(y_true, y_pred, name):
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return {"model": name, "mae": np.nan, "rmse": np.nan, "smape": np.nan,
                "mape": np.nan, "r2": np.nan, "n": int(mask.sum())}
    yt, yp = y_true[mask], y_pred[mask]
    smape = np.mean(2 * np.abs(yp - yt) / (np.abs(yp) + np.abs(yt) + 1e-8)) * 100
    mape_mask = np.abs(yt) >= 1.0  # skip hours near 0 EUR/MWh
    mape = float(np.mean(np.abs((yt[mape_mask] - yp[mape_mask]) / yt[mape_mask])) * 100) \
        if mape_mask.sum() > 0 else np.nan
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "model": name,
        "mae": round(mean_absolute_error(yt, yp), 2),
        "rmse": round(mean_squared_error(yt, yp) ** 0.5, 2),
        "smape": round(smape, 2),
        "mape": round(mape, 2) if np.isfinite(mape) else np.nan,
        "r2": round(r2, 4) if np.isfinite(r2) else np.nan,
        "n": int(len(yt)),
    }


def is_da_column(c):
    return c.startswith("fc_") and ("_da_" in c or c.endswith("_da")) and not ("_id_" in c or c.endswith("_id"))


def is_id_column(c):
    return c.startswith("fc_") and ("_id_" in c or c.endswith("_id"))


def migrate_old_column_names(price_fc):
    """Old caches used fc_arx, fc_lear, fc_dnn_..., new names have _da / _id in them."""
    rename = {}
    for c in price_fc.columns:
        if c == "fc_arx":
            rename[c] = "fc_arx_da"
        elif c.startswith("fc_lear") and "_da" not in c and "_id" not in c:
            base = c[len("fc_"):]
            for model in ("lear_enet", "lear_ridge", "lear"):
                if base.startswith(model):
                    rename[c] = f"fc_{model}_da{base[len(model):]}"
                    break
        elif c.startswith("fc_dnn_") and "_da_" not in c and "_id_" not in c:
            rename[c] = c.replace("fc_dnn_", "fc_dnn_da_", 1)
        elif c.startswith("fc_lstm_") and "_da_" not in c and "_id_" not in c:
            rename[c] = c.replace("fc_lstm_", "fc_lstm_da_", 1)
    return price_fc.rename(columns=rename), rename


class PriceForecast:
    """Holds the data and the forecast table while the model grids run."""

    def __init__(self, cfg, df, df_train, df_test, shap_sets_price, shap_sets_id_price):
        self.cfg = cfg
        self.df = df
        self.df_train = df_train
        self.df_test = df_test
        self.test_year = cfg["test_year"]
        self.out_dir = Path(cfg["out_dir"])
        self.cache_path = self.out_dir / "price_fc.csv"

        modes = cfg["train_mode"]
        self.modes = modes if isinstance(modes, list) else [modes]
        self.primary = self.modes[0]
        # price models run for exactly the configured train modes (no hidden extra mode)
        self.all_modes = self.modes

        self.hgb_params = dict(
            max_iter=int(cfg["hgb_max_iter"]),
            max_depth=int(cfg["hgb_max_depth"]),
            learning_rate=float(cfg["hgb_learning_rate"]),
            min_samples_leaf=int(cfg["hgb_min_samples_leaf"]),
            random_state=int(cfg["hgb_random_state"]),
        )
        self.lstm = {
            "seq": int(cfg["lstm_seq_len_price"]),
            "units": int(cfg["lstm_units"]),
            "epochs": int(cfg["lstm_epochs"]),
            "batch_size": int(cfg["lstm_batch_size"]),
        }

        before = [c for c in (shap_sets_price or {}).get("all", []) if c in df.columns]
        if not before:
            log.warning("no SHAP sets for price, using all non-target columns")
            before = [c for c in df.columns if c not in NON_FEATURES]
        self.price_features = before
        self.arx_features = [c for c in cfg["arx_features_price"] if c in df.columns]
        self.lear_features = [c for c in self.arx_features + ["hour"] if c in df.columns]

        self.shap_sets = {k: [c for c in v if c in df_test.columns] for k, v in (shap_sets_price or {}).items()}
        if not self.shap_sets:
            self.shap_sets = {"all": [c for c in self.price_features if c in df_test.columns]}
        self.shap_sets_id = {k: [c for c in v if c in df_test.columns] for k, v in (shap_sets_id_price or {}).items()}
        if not self.shap_sets_id:
            self.shap_sets_id = self.shap_sets

        self.history = pd.concat([df_train, df_test]).sort_values("datetime").reset_index(drop=True)
        self.meta = {}
        self.price_accuracy = pd.DataFrame()
        self.price_fc = self._load_cache()

    # cache

    def _load_cache(self):
        if not self.cache_path.exists():
            return self.df_test[BASE_COLUMNS].copy()
        price_fc = pd.read_csv(self.cache_path, parse_dates=["datetime"])
        acc_path = self.out_dir / "price_accuracy.csv"
        self.price_accuracy = pd.read_csv(acc_path) if acc_path.exists() else pd.DataFrame()
        price_fc, renamed = migrate_old_column_names(price_fc)
        log.info("resumed from %s", self.cache_path)
        if renamed:
            self.price_fc = price_fc
            self.checkpoint("cache migration")
        return price_fc

    def checkpoint(self, label=""):
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.price_fc.to_csv(self.cache_path, index=False)
            log.info("checkpoint %s: %d columns", label, len(self.price_fc.columns))
        except Exception as e:
            log.warning("checkpoint %s failed: %s", label, e)

    def is_done(self, col):
        """A column counts as done if more than half of it is filled (used to resume from cache)."""
        return col in self.price_fc.columns and self.price_fc[col].notna().mean() > 0.5

    # ARX

    def arx(self, target):
        """24 OLS models, one per hour, trained on df_train."""
        preds = pd.Series(np.nan, index=self.df_test.index)
        n_samples = 0
        feats = self.arx_features
        if len(feats) >= 2:
            for h in range(24):
                train_h = self.df_train[self.df_train["hour"] == h].dropna(subset=feats + [target])
                test_h = self.df_test[self.df_test["hour"] == h].dropna(subset=feats)
                if len(train_h) > 50 and len(test_h) > 0:
                    model = LinearRegression().fit(train_h[feats], train_h[target])
                    preds.loc[test_h.index] = model.predict(test_h[feats])
                    n_samples += len(train_h)
        return preds.values, n_samples

    # LSTM

    def lstm_forecast(self, features, mode, target):
        retrain_step = int(self.cfg["retrain_step"])
        retrain_days = int(self.cfg["lstm_retrain_days"])
        days = forecasting.mode_window_days(mode)
        seq = self.lstm["seq"]
        df_test, history = self.df_test, self.history
        n_test = len(df_test)

        te_X_raw = df_test[features].ffill().bfill().values
        te_y_raw = df_test[target].ffill().bfill().values.reshape(-1, 1)

        br = forecasting.lstm_branches(features, "price")
        preds = []
        n_retrains = 0
        model = None
        t0 = time.perf_counter()

        if mode.startswith("static"):
            t_end = pd.Timestamp(f"{self.test_year}-01-01")
            train = self.df[(self.df["datetime"] >= t_end - pd.Timedelta(days=days)) & (self.df["datetime"] < t_end)]
            tr_X_raw = train[features].ffill().bfill().values
            tr_y_raw = train[target].ffill().bfill().values.reshape(-1, 1)
            model, _, _, _ = forecasting.fit_lstm(tr_X_raw, tr_y_raw, br, self.lstm,
                                                  "price_out", "multi_input_lstm_price")
            n_retrains = 1

            for step in range(0, n_test, retrain_step):
                end = min(step + retrain_step, n_test)
                # scalers see the training data plus all test hours before this chunk
                win_X = np.vstack([tr_X_raw, te_X_raw[:step]]) if step > 0 else tr_X_raw
                win_y = np.vstack([tr_y_raw, te_y_raw[:step]]) if step > 0 else tr_y_raw
                sc_X = MinMaxScaler().fit(win_X)
                sc_y = MinMaxScaler().fit(win_y)
                preds.extend(forecasting.predict_lstm(
                    model, br, sc_y,
                    sc_X.transform(win_X[-seq:]), sc_y.transform(win_y[-seq:]).ravel(),
                    sc_X.transform(te_X_raw[step:end]), sc_y.transform(te_y_raw[step:end]).ravel(),
                    seq))
        else:
            sc_X = sc_y = None
            last_retrain = None
            for step in tqdm(range(0, n_test, retrain_step), desc=f"  LSTM {mode}", ncols=80):
                end = min(step + retrain_step, n_test)
                now = df_test.iloc[step]["datetime"]

                if model is None or last_retrain is None or (now - last_retrain).days >= retrain_days:
                    in_window = (history["datetime"] >= now - pd.Timedelta(days=days)) & (history["datetime"] < now)
                    window = history[in_window]
                    X_win = window[features].ffill().bfill().values
                    y_win = window[target].ffill().bfill().values.reshape(-1, 1)
                    if len(X_win) >= seq + 24:
                        model, sc_X, sc_y, _ = forecasting.fit_lstm(X_win, y_win, br, self.lstm,
                                                                    "price_out", "multi_input_lstm_price")
                        last_retrain = now
                        n_retrains += 1

                if model is None:
                    preds.extend([np.nan] * (end - step))
                    continue

                ctx = history[history["datetime"] < now].tail(seq)
                preds.extend(forecasting.predict_lstm(
                    model, br, sc_y,
                    sc_X.transform(ctx[features].ffill().bfill().values),
                    sc_y.transform(ctx[[target]].ffill().bfill().values).ravel(),
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

    def lstm_grid(self, shap_sets, target, prefix):
        for mode in self.all_modes:
            for label, feats in shap_sets.items():
                feats = [c for c in feats if c in self.df_test.columns]
                col = f"{prefix}{mode}_shap_{label}"
                if self.is_done(col):
                    continue
                log.info("LSTM %s (%d features)", col, len(feats))
                try:
                    preds, meta = self.lstm_forecast(feats, mode, target)
                    valid = float(np.mean(np.isfinite(preds)))
                    if valid < 0.5:
                        log.warning("%s: only %.0f%% filled, not saved", col, valid * 100)
                        continue
                    self.price_fc[col] = np.asarray(preds)
                    self.meta[col] = meta
                    self.checkpoint(col)
                except Exception as e:
                    log.warning("LSTM %s failed: %s: %s", col, type(e).__name__, e)

    # LEAR

    def lear_models(self):
        """Fresh Lasso/Ridge/ElasticNet with the alphas from config (refit each retrain)."""
        c = self.cfg
        return {
            "lasso": Lasso(alpha=c["lear_lasso_alpha"], max_iter=int(c["lear_max_iter"]), random_state=SEED),
            "ridge": Ridge(alpha=c["lear_ridge_alpha"], random_state=SEED),
            "enet": ElasticNet(alpha=c["lear_enet_alpha"], l1_ratio=c["lear_enet_l1_ratio"],
                               max_iter=int(c["lear_max_iter"]), random_state=SEED),
        }

    def lear_da(self, mode):
        """Lasso, Ridge and ElasticNet on asinh(DA price), retrained every day on a rolling window."""
        suffix = "" if mode == self.modes[0] else f"_{mode}"
        cols = {"lasso": f"fc_lear_da{suffix}", "ridge": f"fc_lear_ridge_da{suffix}",
                "enet": f"fc_lear_enet_da{suffix}"}
        if all(self.is_done(c) for c in cols.values()):
            return
        # static modes also run as a daily rolling window here, of the static length
        window_days = forecasting.mode_window_days(mode)
        feats = self.lear_features
        preds = {k: pd.Series(np.nan, index=self.df_test.index) for k in cols}
        runtime = {k: 0.0 for k in cols}
        iters = {"lasso": [], "enet": []}
        n_retrains = 0
        n_samples = 0
        history, df_test = self.history, self.df_test

        for d in tqdm(sorted(df_test["date"].unique()), desc=f"  LEAR {mode}", ncols=80):
            ts = pd.Timestamp(d)
            window = history[(history["datetime"] < ts) & (history["datetime"] >= ts - pd.Timedelta(days=window_days))
                             ].dropna(subset=feats + ["dayahead_price"])
            day = df_test[df_test["date"] == d].dropna(subset=feats)
            if len(window) <= 500 or len(day) == 0:
                continue
            y = np.arcsinh(window["dayahead_price"].values)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(window[feats].values)
            X_te = scaler.transform(day[feats].values)
            models = self.lear_models()
            for name, model in models.items():
                t = time.perf_counter()
                model.fit(X_tr, y)
                preds[name].loc[day.index] = np.sinh(model.predict(X_te))
                if name in iters:
                    iters[name].append(getattr(model, "n_iter_", 0))
                runtime[name] += time.perf_counter() - t
            n_retrains += 1
            n_samples = len(window)

        avg = {k: (np.mean(v) if v else 0) for k, v in iters.items()}
        details = {"lasso": ("LEAR_Lasso", f"~{avg['lasso']:.0f} iter/retrain avg (max 20000)"),
                   "ridge": ("LEAR_Ridge", "closed-form (Cholesky)"),
                   "enet": ("LEAR_ElasticNet", f"~{avg['enet']:.0f} iter/retrain avg (max 20000)")}
        for name, col in cols.items():
            valid = preds[name].notna().mean()
            if valid < 0.5:
                log.warning("%s: only %.0f%% filled, not saved", col, valid * 100)
                continue
            self.price_fc[col] = preds[name].values
            key, detail = details[name]
            self.meta[f"{key}{suffix}"] = {"runtime_s": round(runtime[name], 2), "n_samples_train": n_samples,
                                           "n_features": len(feats), "n_retrains": n_retrains,
                                           "epochs_or_iter": detail}
            self.checkpoint(col)

    def lear_id(self, mode):
        """Same as lear_da for the ID price. Static modes use a fixed window here."""
        suffix = "" if mode == self.modes[0] else f"_{mode}"
        cols = [f"fc_lear_id{suffix}", f"fc_lear_ridge_id{suffix}", f"fc_lear_enet_id{suffix}"]
        if all(self.is_done(c) for c in cols):
            return
        feats = self.lear_features
        df_test = self.df_test
        lasso, ridge, enet = (pd.Series(np.nan, index=df_test.index) for _ in range(3))
        if mode.startswith("rolling"):
            window_days = ROLLING_DAYS.get(mode, 365)
            source = self.history
        else:
            years = list(range(self.test_year - STATIC_YEARS.get(mode, 2), self.test_year))
            source = self.history[self.history["year"].isin(years)]

        for d in tqdm(sorted(df_test["date"].unique()), desc=f"  LEAR ID {mode}", ncols=80):
            ts = pd.Timestamp(d)
            if mode.startswith("rolling"):
                window = source[(source["datetime"] < ts) & (source["datetime"] >= ts - pd.Timedelta(days=window_days))]
            else:
                window = source[source["datetime"] < ts]
            window = window.dropna(subset=feats + ["intraday_price"])
            day = df_test[df_test["date"] == d].dropna(subset=feats)
            if len(window) <= 500 or len(day) == 0:
                continue
            y = np.arcsinh(window["intraday_price"].values)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(window[feats].values)
            X_te = scaler.transform(day[feats].values)
            m = self.lear_models()
            lasso.loc[day.index] = np.sinh(m["lasso"].fit(X_tr, y).predict(X_te))
            ridge.loc[day.index] = np.sinh(m["ridge"].fit(X_tr, y).predict(X_te))
            enet.loc[day.index] = np.sinh(m["enet"].fit(X_tr, y).predict(X_te))
        for col, preds in zip(cols, [lasso, ridge, enet]):
            self.price_fc[col] = preds.values
        self.checkpoint(f"LEAR ID {mode}")

    # DNN

    def dnn_forecast(self, mode, features, target):
        """MLP 256-128 on asinh(price), retrained every 7 days on a rolling window."""
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        df_test, history = self.df_test, self.history
        feats = [c for c in features if c in df_test.columns]
        preds = pd.Series(np.nan, index=df_test.index)
        t0 = time.perf_counter()
        n_retrains = 0
        n_samples = 0
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
        for d in tqdm(sorted(df_test["date"].unique()), desc=f"  DNN {mode}", ncols=80):
            ts = pd.Timestamp(d)
            if last_date is None or (ts - pd.Timestamp(last_date)).days >= 7:
                window = history[(history["datetime"] < ts) & (history["datetime"] >= ts - pd.Timedelta(days=days))
                                 ].dropna(subset=[target])
                if len(window) > 500:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(window[feats].ffill().bfill().fillna(0))
                    tf.random.set_seed(SEED)
                    model.set_weights(init_weights)
                    with forecasting.device_ctx():
                        hist = model.fit(X_tr, np.arcsinh(window[target].values), epochs=500, batch_size=64,
                                         validation_split=0.1, verbose=0,
                                         callbacks=[EarlyStopping(patience=10, restore_best_weights=True, verbose=0)])
                    last_date = d
                    n_retrains += 1
                    n_samples = len(window)
                    epochs.append(len(hist.history.get("loss", [])))
                    n_params = int(model.count_params())

            day = df_test[df_test["date"] == d]
            if scaler is not None and len(day) > 0:
                X_pred = scaler.transform(day[feats].ffill().bfill().fillna(0))
                with forecasting.device_ctx():
                    preds.loc[day.index] = np.sinh(model.predict(X_pred, verbose=0).ravel())

        meta = {
            "runtime_s": round(time.perf_counter() - t0, 2),
            "n_samples_train": n_samples,
            "n_features": len(feats),
            "n_retrains": n_retrains,
            "epochs_or_iter": (f"rolling · window={days}d · {n_retrains} retrain(s) · "
                               f"avg {int(np.mean(epochs)) if epochs else 0} epochs · {n_params:,} params"),
        }
        return preds, meta

    def dnn_grid(self, shap_sets, target, prefix):
        for mode in self.all_modes:
            for label, feats in shap_sets.items():
                feats = [c for c in feats if c in self.df_test.columns]
                col = f"{prefix}{mode}_shap_{label}"
                if self.is_done(col):
                    continue
                log.info("DNN %s (%d features)", col, len(feats))
                try:
                    preds, meta = self.dnn_forecast(mode, feats, target)
                    valid = preds.notna().mean()
                    if valid < 0.5:
                        log.warning("%s: only %.0f%% filled, not saved", col, valid * 100)
                        continue
                    self.price_fc[col] = preds.values
                    self.meta[col] = meta
                    self.checkpoint(col)
                except Exception as e:
                    log.warning("DNN %s failed: %s: %s", col, type(e).__name__, e)

    # HGB

    def hgb_forecast(self, features, label, mode, target, retrain_every=7):
        df_test, history = self.df_test, self.history
        t0 = time.perf_counter()
        n_retrains = n_samples = n_iter = 0
        preds = pd.Series(np.nan, index=df_test.index)

        if mode.startswith("rolling"):
            window_days = ROLLING_DAYS.get(mode, 365)
            model = None
            last = None
            for d in tqdm(sorted(df_test["date"].unique()), desc=f"  HGB {label}", ncols=80):
                if model is None or (last and (pd.Timestamp(d) - pd.Timestamp(last)).days >= retrain_every):
                    cutoff = pd.Timestamp(d)
                    train = history[(history["datetime"] >= cutoff - pd.Timedelta(days=window_days)) &
                                    (history["datetime"] < cutoff)].dropna(subset=[target])
                    if len(train) > 200:
                        model = HistGradientBoostingRegressor(**self.hgb_params).fit(train[features], train[target])
                        last = d
                        n_retrains += 1
                        n_samples = len(train)
                        n_iter = getattr(model, "n_iter_", 300)
                if model is not None:
                    day = df_test["date"] == d
                    preds.loc[day] = model.predict(df_test.loc[day, features])
        else:
            years = list(range(self.test_year - STATIC_YEARS.get(mode, 2), self.test_year))
            train = history[history["year"].isin(years)].dropna(subset=[target])
            if len(train) == 0:
                raise ValueError(f"no training rows for mode {mode!r}, years {years}")
            model = HistGradientBoostingRegressor(**self.hgb_params).fit(train[features], train[target])
            preds = pd.Series(model.predict(df_test[features]), index=df_test.index)
            n_retrains = 1
            n_samples = len(train)
            n_iter = getattr(model, "n_iter_", 300)

        rolling = mode.startswith("rolling")
        meta = {
            "runtime_s": round(time.perf_counter() - t0, 2),
            "n_samples_train": n_samples,
            "n_features": len(features),
            "n_retrains": n_retrains,
            "epochs_or_iter": (f"{n_iter} iterations · retrain every {retrain_every}d" if rolling
                               else f"{n_iter} iterations"),
        }
        return preds, meta

    def hgb_grid(self, shap_sets, target, prefix):
        """Rolling modes retrain weekly; static modes train once. Set cfg['price_hgb_daily_retrain']
        to also run a daily-retrain variant (suffix _daily), as in the thesis (much slower)."""
        daily = bool(self.cfg.get("price_hgb_daily_retrain", False))
        for mode in self.all_modes:
            if mode.startswith("rolling"):
                retrain_cfgs = [(7, ""), (1, "_daily")] if daily else [(7, "")]
            else:
                retrain_cfgs = [(7, "")]
            for retrain_every, freq in retrain_cfgs:
                for label, feats in shap_sets.items():
                    feats = [c for c in feats if c in self.df_test.columns]
                    col = f"{prefix}{mode}{freq}_shap_{label}"
                    if self.is_done(col):
                        continue
                    log.info("HGB %s (%d features)", col, len(feats))
                    try:
                        preds, meta = self.hgb_forecast(feats, f"{mode}{freq}/shap_{label}", mode,
                                                        target, retrain_every)
                        valid = pd.Series(preds.values).notna().mean()
                        if valid < 0.5:
                            log.warning("%s: only %.0f%% filled, not saved", col, valid * 100)
                            continue
                        self.price_fc[col] = preds.values
                        self.meta[col] = meta
                        self.checkpoint(col)
                    except Exception as e:
                        log.warning("HGB %s failed: %s: %s", col, type(e).__name__, e)

    def add_extended_lags(self):
        """Add DA and ID price lags 25..167 so HGB sees the same week of prices as the LSTM."""
        full = pd.concat([self.df_train, self.df_test]).sort_values("datetime").reset_index(drop=True)
        series = [("dayahead_price", "dayahead_price_lag"), ("intraday_price", "intraday_price_lag")]
        new_cols = [f"{pfx}{lag}" for _, pfx in series for lag in range(25, 168) if f"{pfx}{lag}" not in full.columns]
        if new_cols:
            lags = pd.DataFrame({f"{pfx}{lag}": full[col].shift(lag)
                                 for col, pfx in series for lag in range(25, 168) if f"{pfx}{lag}" in new_cols},
                                index=full.index)
            n_test = len(self.df_test)
            self.df_test = self.df_test.assign(**{c: lags[c].values[-n_test:] for c in new_cols}).copy()
            self.df_test["date"] = pd.to_datetime(self.df_test["datetime"]).dt.date
            self.history = self.history.assign(**{c: lags[c].values for c in new_cols}).copy()
            log.info("HGB-ext: added %d price lag columns", len(new_cols))

        ext = [f"{pfx}{lag}" for pfx in ("dayahead_price_lag", "intraday_price_lag")
               for lag in range(25, 168) if f"{pfx}{lag}" in self.df_test.columns]

        def extend(sets):
            return {k: [c for c in v if c in self.df_test.columns] + [c for c in ext if c not in v]
                    for k, v in sets.items()}

        return extend(self.shap_sets), extend(self.shap_sets_id)

    # Run everything

    def run(self):
        log.info("price forecast: modes %s (+ rolling_1y), %d features", self.modes, len(self.price_features))
        families = set(self.cfg.get("model_families", ["lear", "dnn", "hgb", "hgb_ext", "lstm"]))

        # DA models
        if not self.is_done("fc_arx_da"):
            t0 = time.perf_counter()
            self.price_fc["fc_arx_da"], n_samples = self.arx("dayahead_price")
            self.meta["ARX"] = {"runtime_s": round(time.perf_counter() - t0, 2), "n_samples_train": n_samples,
                                "n_features": len(self.arx_features), "n_models": 24,
                                "epochs_or_iter": "closed-form OLS (24 hourly LR)"}
            self.checkpoint("ARX")
        if "lstm" in families:
            self.lstm_grid(self.shap_sets, "dayahead_price", "fc_lstm_da_")
        else:
            log.info("LSTM DA grid skipped (not in cfg['model_families'])")
        if "lear" in families:
            for mode in self.all_modes:
                self.lear_da(mode)
        else:
            log.info("LEAR DA skipped (not in cfg['model_families'])")
        if "dnn" in families:
            self.dnn_grid(self.shap_sets, "dayahead_price", "fc_dnn_da_")
        else:
            log.info("DNN DA grid skipped (not in cfg['model_families'])")
        if "hgb" in families:
            self.hgb_grid(self.shap_sets, "dayahead_price", "fc_hgb_da_")
        else:
            log.info("HGB DA grid skipped (not in cfg['model_families'])")
        shap_sets_ext, shap_sets_id_ext = self.add_extended_lags()
        if "hgb_ext" in families:
            self.hgb_grid(shap_sets_ext, "dayahead_price", "fc_hgb_ext_da_")
        else:
            log.info("HGB-ext DA grid skipped (not in cfg['model_families'])")

        # ID models (same families, target intraday_price)
        if not self.is_done("fc_arx_id"):
            self.price_fc["fc_arx_id"], _ = self.arx("intraday_price")
            self.checkpoint("ARX ID")
        if "lear" in families:
            for mode in self.all_modes:
                self.lear_id(mode)
        else:
            log.info("LEAR ID skipped (not in cfg['model_families'])")
        if "hgb" in families:
            self.hgb_grid(self.shap_sets_id, "intraday_price", "fc_hgb_id_")
            self.checkpoint("HGB ID")
        else:
            log.info("HGB ID grid skipped (not in cfg['model_families'])")
        if "hgb_ext" in families:
            self.hgb_grid(shap_sets_id_ext, "intraday_price", "fc_hgb_ext_id_")
            self.checkpoint("HGB-ext ID")
        else:
            log.info("HGB-ext ID grid skipped (not in cfg['model_families'])")
        if "dnn" in families:
            self.dnn_grid(self.shap_sets_id, "intraday_price", "fc_dnn_id_")
            self.checkpoint("DNN ID")
        else:
            log.info("DNN ID grid skipped (not in cfg['model_families'])")
        if "lstm" in families:
            self.lstm_grid(self.shap_sets_id, "intraday_price", "fc_lstm_id_")
            self.checkpoint("LSTM ID")
        else:
            log.info("LSTM ID grid skipped (not in cfg['model_families'])")

        fc = self.price_fc
        actual_da = fc["dayahead_price"].values
        actual_id = fc["intraday_price"].values
        id_cols = [c for c in fc.columns if is_id_column(c) and fc[c].notna().sum() > 100]
        id_accuracy = pd.DataFrame([accuracy(actual_id, fc[c].values, c) for c in id_cols]
                                   ).sort_values("mae").reset_index(drop=True)

        # Static market rule: on average cheaper market per (hour, holiday) in the training window
        market_rule = {}
        market_accuracy = pd.DataFrame(columns=["model", "mkt_acc"])
        price_accuracy = self.price_accuracy
        has_forecasts = any(c.startswith("fc_") and not c.startswith("fc_spread_") and fc[c].notna().mean() > 0.5
                            for c in fc.columns)
        if has_forecasts:
            rule = self.df_train.groupby(["hour", "is_holiday"]).agg(
                mean_DA=("dayahead_price", "mean"), mean_ID=("intraday_price", "mean"))
            rule["market"] = np.where(rule["mean_DA"] <= rule["mean_ID"], "DA", "ID")
            market_rule = rule["market"].to_dict()
            fc["market_static"] = fc.apply(
                lambda r: market_rule.get((int(r["hour"]), bool(r["is_holiday"])), "DA"), axis=1)

            da_cols = [c for c in fc.columns if is_da_column(c)]
            price_accuracy = pd.DataFrame([accuracy(actual_da, fc[c].values, c) for c in da_cols]).sort_values("mae")

            # Routing accuracy: forecast DA vs actual ID -> chosen market, compared with the cheaper one ex post
            actual_market = np.where(fc["dayahead_price"] <= fc["intraday_price"], "DA", "ID")
            rows = []
            for c in da_cols:
                values = fc[c].values
                implied = np.where(values <= actual_id, "DA", "ID")
                mask = np.isfinite(values)
                acc = (implied[mask] == actual_market[mask]).mean() if mask.sum() > 0 else np.nan
                rows.append({"model": c, "mkt_acc": round(acc * 100, 1) if np.isfinite(acc) else np.nan})
            market_accuracy = pd.DataFrame(rows).sort_values("mkt_acc", ascending=False)
            static_acc = (fc["market_static"].values == actual_market).mean()
            log.info("static rule routing accuracy: %.1f%%", static_acc * 100)

        self.checkpoint("final")
        try:
            price_accuracy.to_csv(self.out_dir / "price_accuracy.csv", index=False)
            if not id_accuracy.empty:
                id_accuracy.to_csv(self.out_dir / "id_accuracy.csv", index=False)
            if self.meta:
                pd.DataFrame(self.meta).T.reset_index().rename(columns={"index": "model"}).to_csv(
                    self.out_dir / "training_meta.csv", index=False)
        except Exception as e:
            log.warning("saving failed: %s", e)

        log.info("DA accuracy:\n%s", price_accuracy.to_string(index=False))
        log.info("ID accuracy:\n%s", id_accuracy.to_string(index=False))
        return {
            "price_fc": fc,
            "price_accuracy": price_accuracy,
            "id_accuracy": id_accuracy,
            "market_accuracy": market_accuracy,
            "market_rule_train": market_rule,
            "price_training_meta": self.meta,
        }


def run(cfg, df, df_train, df_test, shap_sets_price, shap_sets_id_price):
    # Keras and scikit-learn print many UserWarnings in the grids, mute them only here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        families = set(cfg.get("model_families", ["lear", "dnn", "hgb", "hgb_ext", "lstm"]))
        if "lstm" in families or "dnn" in families:
            forecasting.set_tf_device(cfg.get("tf_device", "/CPU:0"))
            if cfg["seed"] is not None:
                import tensorflow as tf

                tf.keras.utils.set_random_seed(cfg["seed"])
        return PriceForecast(cfg, df, df_train, df_test, shap_sets_price, shap_sets_id_price).run()
