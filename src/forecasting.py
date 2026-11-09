"""Shared forecasting blocks for steps 4 and 5: feature groups, LSTM/DNN builders and
window helpers. TF and sklearn are imported inside the functions that use them."""
import contextlib
import logging
import re

import numpy as np

log = logging.getLogger(__name__)

# Device the neural nets run on. CPU by default: GPU LSTM training is often
# non-deterministic even with a seed, so CPU keeps runs reproducible. Set via
# set_tf_device(cfg["tf_device"]); "auto"/"" lets TensorFlow choose (e.g. a GPU).
_TF_DEVICE = "/CPU:0"


def set_tf_device(name):
    global _TF_DEVICE
    _TF_DEVICE = name or "auto"


def device_ctx():
    """Placement context for the Keras models: CPU by default, TF's choice on 'auto'."""
    if not _TF_DEVICE or _TF_DEVICE.lower() == "auto":
        return contextlib.nullcontext()
    import tensorflow as tf
    return tf.device(_TF_DEVICE)

# feature groups for the LSTM branch split
CALENDAR_FEATURES = {"hour", "weekday", "month", "year", "is_weekend", "is_holiday",
                     "is_bridge_day", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                     "doy_sin", "doy_cos", "peak_hour", "peak_day", "peak_month"}
LOAD_PREFIXES = ("load_lag", "load_rmean", "load_rstd")
PRICE_PREFIXES = ("dayahead_price_lag", "dayahead_price_rmean", "dayahead_price_rstd",
                  "intraday_price_lag", "intraday_price_rmean", "intraday_price_rstd",
                  "spread_lag", "da_price_es_lag", "id_price_es_lag",
                  "spread_es_pt_lag", "spread_id_es_pt_lag")
MARKET_PREFIXES = ("ttf_lag", "eua_lag")
GENERATION_PREFIXES = ("wind_lag", "solar_lag", "gen_gas_lag", "gen_coal_lag",
                       "hydro_ror_lag", "flow_net_MW_lag", "wind_fc_lag", "solar_fc_lag", "congestion_lag")


def mode_window_days(mode):
    """static_2y -> 730, rolling_6m -> 180, rolling_1y -> 365. Raises on an unknown mode."""
    m = re.match(r"(static|rolling)_(\d+)(y|m)$", mode.lower())
    if not m:
        raise ValueError(f"Unknown train_mode {mode!r}. Use static_Xy or rolling_Xm / rolling_Xy.")
    n, unit = int(m.group(2)), m.group(3)
    return n * 365 if unit == "y" else n * 30


def lstm_branches(features, target_kind):
    # target_kind "load" or "price" picks which lags form the target branch;
    # the other kind becomes the extra branch. Pick order and branch order match
    # the originals in steps 4 and 5, so the models stay identical.
    if target_kind == "load":
        target_pred = lambda c: c.startswith(LOAD_PREFIXES)
        extra_name, extra_pred = "price", lambda c: c.startswith(PRICE_PREFIXES)
    elif target_kind == "price":
        target_pred = lambda c: c.startswith(PRICE_PREFIXES)
        extra_name, extra_pred = "load", lambda c: c.startswith(LOAD_PREFIXES)
    else:
        raise ValueError(f"target_kind must be 'load' or 'price', got {target_kind!r}")

    used = set()

    def pick(pred):
        cols = [c for c in features if pred(c) and c not in used]
        used.update(cols)
        return cols

    groups = {
        "time": pick(lambda c: c in CALENDAR_FEATURES),
        "target": pick(target_pred),
        extra_name: pick(extra_pred),
        "market": pick(lambda c: c.startswith(MARKET_PREFIXES)),
        "genfund": pick(lambda c: c.startswith(GENERATION_PREFIXES)),
        "weather": pick(lambda c: c.startswith("era5_")),
    }
    for fixed in ("time", "target"):
        if not groups[fixed]:
            groups[fixed] = features[:1]
            log.warning("LSTM branch %r is empty, using a fallback feature", fixed)

    branches = ["time", "target"] + [b for b in (extra_name, "market", "genfund", "weather") if groups[b]]
    result = {"branches": branches, "target_branch": "target"}
    for b in branches:
        result[f"{b}_feat"] = groups[b]
        result[f"{b}_idx"] = [features.index(c) for c in groups[b] if c in features]
    return result


def make_sequences(X, y, seq_len):
    """Sliding windows: X[i-seq_len:i] -> y[i]."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def split_inputs(X3d, br):
    return {f"{name}_input": X3d[:, :, br[f"{name}_idx"]] for name in br["branches"]}


def build_lstm(seq_len, br, units, out_name="output", model_name="multi_input_lstm"):
    """One LSTM per branch, concatenated, then Dense -> Dropout -> output."""
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Concatenate, Dense, Dropout, Input
    from tensorflow.keras.models import Model

    with device_ctx():
        inputs, outputs = [], []
        for name in br["branches"]:
            n_feat = len(br[f"{name}_idx"])
            size = units if name == "target" else units // 2
            inp = Input(shape=(seq_len, n_feat), name=f"{name}_input")
            inputs.append(inp)
            outputs.append(LSTM(size, name=f"{name}_lstm")(inp))

        x = Concatenate()(outputs) if len(outputs) > 1 else outputs[0]
        x = Dense(units * 2, activation="relu")(x)
        x = Dropout(0.2)(x)
        out = Dense(1, name=out_name)(x)

        model = Model(inputs=inputs, outputs=out, name=model_name)
        model.compile(optimizer="adam", loss="mse")
    return model


def fit_lstm(X_raw, y_raw, br, lstm, out_name="output", model_name="multi_input_lstm"):
    """Scale X and y with MinMaxScaler, build the branch LSTM and fit with early stopping."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler

    sc_X = MinMaxScaler().fit(X_raw)
    sc_y = MinMaxScaler().fit(y_raw)
    Xt, yt = make_sequences(sc_X.transform(X_raw), sc_y.transform(y_raw).ravel(), lstm["seq"])
    model = build_lstm(lstm["seq"], br, lstm["units"], out_name, model_name)
    with device_ctx():
        hist = model.fit(
            split_inputs(Xt, br), yt,
            epochs=lstm["epochs"],
            batch_size=lstm["batch_size"],
            validation_split=0.1,
            shuffle=False,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True, verbose=0)],
            verbose=0,
        )
    n_epochs = len(hist.history["loss"])
    log.info("    LSTM fitted: %d epochs (max %d)", n_epochs, lstm["epochs"])
    return model, sc_X, sc_y, n_epochs


def predict_lstm(model, br, sc_y, ctx_X, ctx_y, chunk_X, chunk_y, seq_len):
    # prepend the context window, predict the chunk, inverse-scale y; inputs are already scaled
    import tensorflow as tf

    X_chunk, _ = make_sequences(np.vstack([ctx_X, chunk_X]), np.concatenate([ctx_y, chunk_y]), seq_len)
    with device_ctx():
        pred = model.predict(split_inputs(X_chunk, br), verbose=0).flatten()
    return sc_y.inverse_transform(pred.reshape(-1, 1)).ravel().tolist()


def build_dnn(n_features, lr=0.001):
    """MLP 256-128-1 on the CPU, Adam + MSE. Identical in steps 4 and 5."""
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    with device_ctx():
        model = Sequential([
            Dense(256, activation="relu", input_shape=(n_features,)),
            Dense(128, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model
