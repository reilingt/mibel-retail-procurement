"""Step 1: build the feature table from the merged hourly CSV, split train/test, rank with SHAP.
Returns df, df_train, df_test and the SHAP feature sets for steps 4 and 5."""
import logging
import os
import tempfile
from datetime import timedelta

import holidays
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RAW_CSV = "merged_2018_2024_hourly_final.csv"
CLEAN_CSV = "merged_2018_2024_hourly_final_clean.csv"
ERA5_CACHE = os.path.join("cache", "era5_weather_v2.npz")
GRIB_DIR = "datasets"
GRIB_FILES = ["era5_u100.grib", "era5_v100.grib", "era5_tp.grib", "era5_t2m.grib"]

# Day-ahead gate closure is D-1 12:00, so every lagged input is at least 24 h old
LAGS = [24, 168]

ERA5_BBOX = {"lat_min": 36.5, "lat_max": 43.0, "lon_min": -9.5, "lon_max": -6.0}
# wind_norte: Minho onshore wind area, tp_norte: Douro/Minho hydro basins,
# t2m_*: nearest grid point to each city. Solar radiation is not used: the ENTSO-E solar
# forecast carries that information and the ERA5 radiation series did not line up in time
ERA5_ZONES = {
    "wind_norte": {"type": "box", "lat": (41.5, 42.5), "lon": (-8.5, -7.5)},
    "tp_norte": {"type": "box", "lat": (41.0, 42.5), "lon": (-8.0, -6.5)},
    "t2m_porto": {"type": "point", "lat": 41.15, "lon": -8.61},
    "t2m_lisboa": {"type": "point", "lat": 38.72, "lon": -9.14},
    "t2m_sul": {"type": "point", "lat": 37.02, "lon": -7.93},
}

# Recreated later, replaced by ERA5, or not used by any model
DROP_COLS = [
    "hour", "weekday", "month", "year",
    "precip_mm", "temp_era5_celsius", "wind_speed_u10",
    "gas_marginal_cost", "wind_curtail_pct",
    "wind_forecast_error_mw", "solar_forecast_error_mw",
    "loss", "heating_deg_days", "cooling_deg_days",
    "peak_hour", "peak_day", "peak_month",
    "residual_demand_mw", "renewable_pct", "net_imports_mw",
]

RENAME = {
    "da_price": "dayahead_price",
    "id_price": "intraday_price",
    "demand_mw": "load",
    "net_flow_mw": "flow_net_MW",
    "ttf_eur_mwh": "ttf",
    "carbon_eur_t": "eua",
    "gen_wind_mw": "wind",
    "gen_solar_mw": "solar",
    "gen_hydro_ror_mw": "hydro_ror",
    "congestion_ratio": "congestion",
    "ntc_pt_es_mw": "ntc_pt_to_es",
    "ntc_es_pt_mw": "ntc_es_to_pt",
    "gen_gas_mw": "gen_gas",
    "gen_coal_mw": "gen_coal",
    "wind_forecast_mw": "wind_fc",
    "solar_forecast_mw": "solar_fc",
}

FUNDAMENTALS = [
    "wind", "solar", "flow_net_MW", "ttf", "eua",
    "gen_gas", "gen_coal", "hydro_ror", "congestion",
    "wind_fc", "solar_fc",
]
# Raw same-hour values without lag versions, not known at day-ahead time
RAW_NO_LAG = [
    "gen_hydro_pump_mw", "gen_hydro_res_mw", "gen_biomass_mw",
    "wind_speed_north", "ntc_es_to_pt", "ntc_pt_to_es",
]

CALENDAR_FEATURES = {
    "hour", "weekday", "month", "year", "is_weekend", "is_holiday",
    "is_bridge_day", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "doy_sin", "doy_cos", "peak_hour", "peak_day", "peak_month",
}

# Columns that never go into SHAP: identifiers, targets and same-hour fundamentals
# (wind_fc / solar_fc stay in, they are published the day before)
NON_FEATURES = {
    "datetime", "date", "year", "dayofyear", "holiday_name",
    "load", "dayahead_price", "intraday_price",
    "wind", "solar", "flow_net_MW", "ttf", "eua",
    "gen_gas", "gen_coal", "hydro_ror", "congestion",
    *RAW_NO_LAG,
}
# lag-1 features are never built (LAGS starts at 24), so load and price share one list
NON_FEATURES_LOAD = NON_FEATURES
NON_FEATURES_PRICE = NON_FEATURES


# ERA5 weather

def _era5_load_field(path):
    """Load one GRIB file clipped to Portugal -> (values, time, lat, lon)."""
    import xarray as xr

    os.environ.setdefault("GRIB_CACHE_DIR", tempfile.gettempdir())
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})

    lat_n = "latitude" if "latitude" in ds.coords else "lat"
    lon_n = "longitude" if "longitude" in ds.coords else "lon"
    lon_vals = np.asarray(ds[lon_n].values, float)
    if np.nanmax(lon_vals) > 180.0:
        lon_vals = ((lon_vals + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords({lon_n: lon_vals})
    lat_vals = np.asarray(ds[lat_n].values, float)
    lat_lo, lat_hi = ERA5_BBOX["lat_min"], ERA5_BBOX["lat_max"]
    lon_lo, lon_hi = ERA5_BBOX["lon_min"], ERA5_BBOX["lon_max"]
    lat_sl = slice(lat_hi, lat_lo) if lat_vals[0] > lat_vals[-1] else slice(lat_lo, lat_hi)
    lon_sl = slice(lon_hi, lon_lo) if lon_vals[0] > lon_vals[-1] else slice(lon_lo, lon_hi)
    sub = ds.sel({lat_n: lat_sl, lon_n: lon_sl})

    vname = next(v for v in sub.data_vars if str(v).lower() not in ("number", "surface"))
    arr = np.asarray(sub[vname].values, float)

    tv = sub["valid_time"].values if "valid_time" in sub.coords else sub["time"].values
    if isinstance(tv, np.ndarray) and tv.ndim > 1:
        if arr.ndim == 4 and tv.shape == arr.shape[:2]:
            t = pd.DatetimeIndex(pd.to_datetime(tv.reshape(-1)))
            arr = arr.reshape(-1, arr.shape[2], arr.shape[3])
        else:
            t = pd.DatetimeIndex(pd.to_datetime(tv[:, 0]))
    else:
        t = pd.DatetimeIndex(pd.to_datetime(tv))
    t = t.tz_localize("UTC") if getattr(t, "tz", None) is None else t.tz_convert("UTC")

    lat = np.asarray(sub[lat_n].values, float).ravel()
    lon = np.asarray(sub[lon_n].values, float).ravel()
    return arr, t, lat, lon


def _era5_zone_mean(arr, lat, lon, zone):
    if zone["type"] == "point":
        i = int(np.argmin(np.abs(lat - zone["lat"])))
        j = int(np.argmin(np.abs(lon - zone["lon"])))
        return arr[:, i, j].copy()
    lat_idx = np.where((lat >= zone["lat"][0]) & (lat <= zone["lat"][1]))[0]
    lon_idx = np.where((lon >= zone["lon"][0]) & (lon <= zone["lon"][1]))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return np.full(arr.shape[0], np.nan)
    sub = arr[:, lat_idx, :][:, :, lon_idx].reshape(arr.shape[0], -1)
    return np.nanmean(sub, axis=1)


def _era5_align(era_time, values, df_index):
    """Put an ERA5 series on the hourly index of the CSV and fill small gaps."""
    idx = era_time.tz_convert(None) if getattr(era_time, "tz", None) else pd.DatetimeIndex(era_time)
    s = pd.Series(values.astype(float), index=idx)
    s = s[~s.index.duplicated(keep="first")]
    target = pd.DatetimeIndex(df_index)
    if getattr(target, "tz", None) is not None:
        target = target.tz_convert(None)
    return s.reindex(target).interpolate(limit_direction="both").ffill().bfill().values.astype(float)


def _hourly_accumulation(arr):
    # ERA5 tp in these files are already hourly sums, so they are used as-is;
    # only clip the odd negative value from interpolation
    return np.where(np.isfinite(arr) & (arr < 0), 0.0, arr)


def _era5_load_zones(grib_dir):
    def field(name):
        return _era5_load_field(os.path.join(grib_dir, name))

    log.info("ERA5: loading wind")
    u_arr, u_t, u_lat, u_lon = field("era5_u100.grib")
    v_arr, _, _, _ = field("era5_v100.grib")
    wind_norte = _era5_zone_mean(np.sqrt(u_arr**2 + v_arr**2), u_lat, u_lon, ERA5_ZONES["wind_norte"])

    log.info("ERA5: loading precipitation")
    tp_arr, tp_t, tp_lat, tp_lon = field("era5_tp.grib")
    tp_norte = _era5_zone_mean(_hourly_accumulation(tp_arr), tp_lat, tp_lon, ERA5_ZONES["tp_norte"])

    log.info("ERA5: loading temperature")
    t2m_arr, t2m_t, t2m_lat, t2m_lon = field("era5_t2m.grib")
    t2m_c = t2m_arr - 273.15

    return {
        "wind_norte": (wind_norte, u_t),
        "tp_norte": (tp_norte * 1000.0, tp_t),     # m/h -> mm/h
        "t2m_porto": (_era5_zone_mean(t2m_c, t2m_lat, t2m_lon, ERA5_ZONES["t2m_porto"]), t2m_t),
        "t2m_lisboa": (_era5_zone_mean(t2m_c, t2m_lat, t2m_lon, ERA5_ZONES["t2m_lisboa"]), t2m_t),
        "t2m_sul": (_era5_zone_mean(t2m_c, t2m_lat, t2m_lon, ERA5_ZONES["t2m_sul"]), t2m_t),
    }


def _era5_zones_cached(grib_dir, cache_path):
    """ERA5 zone series from the npz cache, or from the GRIB files if the cache is older."""
    cache_valid = False
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        cache_valid = all(
            os.path.getmtime(os.path.join(grib_dir, f)) <= mtime
            for f in GRIB_FILES
            if os.path.exists(os.path.join(grib_dir, f))
        )
    if cache_valid:
        try:
            data = np.load(cache_path, allow_pickle=False)
            zones = {}
            for key in ERA5_ZONES:
                arr = data[f"{key}_arr"].astype(float)
                t = pd.DatetimeIndex(pd.to_datetime(data[f"{key}_time"].astype("datetime64[ns]"))).tz_localize("UTC")
                zones[key] = (arr, t)
            log.info("ERA5: loaded from cache")
            return zones
        except Exception as e:
            log.warning("ERA5 cache unreadable (%s), reading GRIB files", e)

    zones = _era5_load_zones(grib_dir)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        kw = {}
        for key, (arr, t) in zones.items():
            kw[f"{key}_arr"] = arr.astype(np.float32)
            kw[f"{key}_time"] = t.asi8
        np.savez_compressed(cache_path, **kw)
        log.info("ERA5: cache saved to %s", cache_path)
    except Exception as e:
        log.warning("ERA5 cache could not be saved: %s", e)
    return zones


def _shift(arr, lag):
    if lag <= 0:
        return arr.astype(float)
    out = np.full(len(arr), np.nan)
    out[lag:] = arr[:-lag]
    return out


def add_era5_weather_features(df, grib_dir, cache_path):
    """Lagged ERA5 columns: wind, solar and temperature 1 h, precipitation 14 and 21 days."""
    try:
        zones = _era5_zones_cached(grib_dir, cache_path)
    except Exception as e:
        log.warning("ERA5 data not available (%s), skipping weather features", e)
        return df

    def aligned(key):
        return _era5_align(zones[key][1], zones[key][0], pd.DatetimeIndex(df["datetime"]))

    df["era5_wind_norte_l1"] = _shift(aligned("wind_norte"), 1)
    tp = aligned("tp_norte")
    df["era5_tp_norte_l336"] = _shift(tp, 336)  # reservoirs react with a delay
    df["era5_tp_norte_l504"] = _shift(tp, 504)
    for key in ["t2m_porto", "t2m_lisboa", "t2m_sul"]:
        df[f"era5_{key}_l1"] = _shift(aligned(key), 1)

    log.info("ERA5: added %d weather columns", sum(c.startswith("era5_") for c in df.columns))
    return df


# Cleaning

def drop_redundant_columns(df):
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        df = df.drop(columns=all_nan)
        log.info("dropped all-NaN columns: %s", all_nan)
    return df


def normalize_column_names(df):
    """Rename raw CSV columns to the names used in the pipeline (never overwrites)."""
    renamed = {src: dst for src, dst in RENAME.items() if src in df.columns and dst not in df.columns}
    if renamed:
        df = df.rename(columns=renamed)
    return df, renamed


def prepare_source_csv(data_dir):
    """Write the clean CSV once (dropped columns, new names, ERA5). Skipped if it is up to date."""
    raw_path = os.path.join(data_dir, RAW_CSV)
    clean_path = os.path.join(data_dir, CLEAN_CSV)
    if os.path.exists(clean_path) and os.path.getmtime(clean_path) >= os.path.getmtime(raw_path):
        log.info("clean CSV is up to date: %s", clean_path)
        return

    df = pd.read_csv(raw_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df, _ = normalize_column_names(df)
    df = add_era5_weather_features(df, os.path.join(data_dir, GRIB_DIR), os.path.join(data_dir, ERA5_CACHE))
    df.to_csv(clean_path, index=False)
    log.info("clean CSV saved: %s %s", clean_path, df.shape)


def load_data(data_dir):
    clean_path = os.path.join(data_dir, CLEAN_CSV)
    path = clean_path if os.path.exists(clean_path) else os.path.join(data_dir, RAW_CSV)
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    # the DST fall-back hour (last Sunday of October) appears twice in local time;
    # keep the first so datetime stays a unique key for later merges
    dups = df["datetime"].duplicated().sum()
    if dups:
        df = df.drop_duplicates(subset="datetime", keep="first").reset_index(drop=True)
        log.info("dropped %d duplicate timestamp(s) (DST fall-back)", dups)
    return df


# Features

def add_calendar_features(df):
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["date"] = df["datetime"].dt.date
    df["is_weekend"] = df["weekday"].isin([5, 6])
    df["dayofyear"] = df["datetime"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def add_holiday_features(df):
    """Portuguese holidays; a bridge day is a Friday after or a Monday before a holiday."""
    pt_holidays = holidays.country_holidays("PT", years=sorted(df["year"].unique()))
    df["is_holiday"] = df["date"].apply(lambda d: d in pt_holidays)
    df["holiday_name"] = df["date"].apply(lambda d: pt_holidays.get(d, ""))

    holiday_dates = set(d for d in df["date"].unique() if d in pt_holidays)

    def is_bridge(d):
        if d in holiday_dates:
            return False
        if d.weekday() == 4:
            return (d - timedelta(days=1)) in holiday_dates
        if d.weekday() == 0:
            return (d + timedelta(days=1)) in holiday_dates
        return False

    df["is_bridge_day"] = df["date"].map({d: is_bridge(d) for d in df["date"].unique()})
    return df


def add_lag_features(df):
    for col in ["load", "dayahead_price", "intraday_price"]:
        if col in df.columns:
            for lag in LAGS:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    fund_created = []
    for col in FUNDAMENTALS:
        if col in df.columns:
            for lag in LAGS:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
                fund_created.append(f"{col}_lag{lag}")

    missing = [c for c in ["load", "dayahead_price", "intraday_price"] if c not in df.columns]
    if missing:
        log.warning("missing base columns for lags: %s", missing)
    return df, fund_created


def add_spanish_features(df, es_col, pt_col, spread_name):
    """Lags of the Spanish price and of the ES-PT spread. The raw columns are dropped."""
    if es_col not in df.columns:
        return df, False
    for lag in LAGS:
        df[f"{es_col}_lag{lag}"] = df[es_col].shift(lag)
    if pt_col in df.columns:
        spread = df[es_col] - df[pt_col]
        for lag in LAGS:
            df[f"{spread_name}_lag{lag}"] = spread.shift(lag)
    return df.drop(columns=[es_col]), True


def add_rolling_features(df):
    # Windows end at t-24 so they only contain information known at gate closure
    if "load" in df.columns:
        for w in [24, 168]:
            df[f"load_rmean{w}"] = df["load"].shift(24).rolling(w, min_periods=1).mean()
            df[f"load_rstd{w}"] = df["load"].shift(24).rolling(w, min_periods=1).std()
    for col in ["dayahead_price", "intraday_price"]:
        if col in df.columns:
            for w in [24, 168]:
                df[f"{col}_rmean{w}"] = df[col].shift(24).rolling(w, min_periods=1).mean()
                df[f"{col}_rstd{w}"] = df[col].shift(24).rolling(w, min_periods=1).std()
    return df


def add_spread_features(df):
    if "intraday_price" in df.columns and "dayahead_price" in df.columns:
        spread = df["intraday_price"] - df["dayahead_price"]
        for lag in LAGS:
            df[f"spread_lag{lag}"] = spread.shift(lag)
    else:
        for lag in LAGS:
            df[f"spread_lag{lag}"] = np.nan
        log.warning("spread features skipped, price columns missing")
    return df


def build_feature_table(data_dir):
    prepare_source_csv(data_dir)
    df = load_data(data_dir)

    # Already done in the clean CSV, kept as a safety check
    df = drop_redundant_columns(df)
    df, _ = normalize_column_names(df)

    df = add_calendar_features(df)
    df = add_holiday_features(df)
    if not any(c.startswith("era5_") for c in df.columns):
        df = add_era5_weather_features(df, os.path.join(data_dir, GRIB_DIR), os.path.join(data_dir, ERA5_CACHE))

    df, fund_created = add_lag_features(df)
    df, es_loaded = add_spanish_features(df, "da_price_es", "dayahead_price", "spread_es_pt")
    df, es_id_loaded = add_spanish_features(df, "id_price_es", "intraday_price", "spread_id_es_pt")
    df = add_rolling_features(df)
    df = add_spread_features(df)
    return df, fund_created, es_loaded, es_id_loaded


def drop_all_nan_columns(df):
    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        log.warning("dropping %d all-NaN columns, check the data source: %s", len(all_nan), all_nan)
        df = df.drop(columns=all_nan)
    return df


# Train/test split

def get_train_mask(df, target_date, mode, test_year):
    """Rows used for training. target_date only matters for rolling modes."""
    if mode == "static_2y":
        return df["year"].isin([test_year - 2, test_year - 1])
    if mode == "static_1y":
        return df["year"] == (test_year - 1)
    days = {"rolling_12m": 365, "rolling_6m": 182, "rolling_3m": 91}[mode]
    cutoff = pd.Timestamp(target_date) - pd.Timedelta(days=days)
    return (df["datetime"] >= cutoff) & (df["datetime"] < pd.Timestamp(target_date))


def get_train_test_split(df, cfg):
    mode = cfg["train_mode"]
    if isinstance(mode, list):
        mode = mode[0]  # first mode defines df_train
    test_year = cfg["test_year"]
    target_date = None if mode.startswith("static") else pd.Timestamp(f"{test_year}-01-01")
    df_train = df[get_train_mask(df, target_date, mode, test_year)].copy()
    df_test = df[df["year"] == test_year].copy()
    return df_train, df_test


# Summary

def feature_group(col):
    if col in CALENDAR_FEATURES:
        return "calendar"
    if col.startswith(("load_lag", "load_rmean", "load_rstd")):
        return "load"
    if col.startswith(("dayahead_price_lag", "dayahead_price_rmean", "dayahead_price_rstd",
                       "intraday_price_lag", "intraday_price_rmean", "intraday_price_rstd",
                       "spread_lag", "da_price_es_lag", "id_price_es_lag",
                       "spread_es_pt_lag", "spread_id_es_pt_lag")):
        return "market"
    if col.startswith(("ttf_lag", "eua_lag")):
        return "resources"
    if col.startswith(("wind_lag", "solar_lag", "gen_gas_lag", "gen_coal_lag",
                       "hydro_ror_lag", "flow_net_MW_lag", "wind_fc_lag",
                       "solar_fc_lag", "congestion_lag")):
        return "generation"
    if col.startswith("era5_"):
        return "weather"
    return "other"


def log_summary(df, df_train, df_test, cfg):
    log.info("rows: %d, period %s to %s, columns: %d",
             len(df), df["datetime"].min().date(), df["datetime"].max().date(), len(df.columns))
    log.info("train: %d rows (%s), test %s: %d rows, %d holidays",
             len(df_train), cfg["train_mode"], cfg["test_year"], len(df_test),
             df_test.loc[df_test["is_holiday"], "date"].nunique())

    skip = {"datetime", "date", "year", "dayofyear", "holiday_name",
            "load", "dayahead_price", "intraday_price", "congestion", *RAW_NO_LAG}
    groups = {}
    for c in df.columns:
        if c not in skip:
            groups.setdefault(feature_group(c), []).append(c)
    for name in ["calendar", "load", "market", "resources", "generation", "weather", "other"]:
        cols = groups.get(name, [])
        if cols:
            non_nan = df[cols].notna().mean(axis=0).mean() * 100
            log.info("  %-11s %3d features, %5.1f%% non-NaN", name, len(cols), non_nan)

    raw_left = [c for c in FUNDAMENTALS + RAW_NO_LAG if c in df.columns]
    if raw_left:
        log.info("same-hour columns still in df (must not be model features): %s", raw_left)


# SHAP ranking

def shap_rank_features(df_train, df_test, target, non_features, cfg):
    """Fit a reference HGB on an early slice of df_train, rank features by mean |SHAP| on a
    held-out chronological tail of df_train, and cut at cumulative thresholds.

    Both the reference fit and the SHAP explanation stay inside df_train: the feature
    selection never sees df_test (the test year), so it is fully out of sample.
    """
    import shap
    from sklearn.ensemble import HistGradientBoostingRegressor

    features = [c for c in df_train.columns if c in df_test.columns and c not in non_features]
    train_full = df_train.dropna(subset=[target]).sort_values("datetime")

    # Chronological 80/20 split of df_train: fit on the early part, explain on the tail.
    # Falls back to using all of df_train for both if there isn't enough data to split.
    cut = int(len(train_full) * 0.8)
    fit_rows = train_full.iloc[:cut] if cut >= 200 else train_full
    val_rows = train_full.iloc[cut:] if cut >= 200 and len(train_full) - cut >= 50 else train_full

    model = HistGradientBoostingRegressor(
        max_iter=int(cfg["hgb_max_iter"]),
        max_depth=int(cfg["hgb_max_depth"]),
        learning_rate=float(cfg["hgb_learning_rate"]),
        min_samples_leaf=int(cfg["hgb_min_samples_leaf"]),
        random_state=int(cfg["hgb_random_state"]),
    )
    log.info("SHAP %s: fitting reference HGB on %d features (%d train-slice rows)",
             target, len(features), len(fit_rows))
    model.fit(fit_rows[features], fit_rows[target])

    explanation = shap.TreeExplainer(model)(val_rows[features])
    ranked = pd.Series(np.abs(explanation.values).mean(axis=0), index=features).sort_values(ascending=False)
    cumsum = (ranked.cumsum() / ranked.sum()).values

    sets = {}
    for thr in cfg["shap_thresholds"]:
        if thr >= 1.0:
            sets["all"] = ranked.index.tolist()
        else:
            n = int(np.searchsorted(cumsum, thr)) + 1
            sets[f"p{int(thr * 100)}"] = ranked.index[:min(n, len(ranked))].tolist()
    for label, feats in sets.items():
        log.info("  %5s: %3d features", label, len(feats))
    log.info("SHAP %s top 5: %s", target, ", ".join(ranked.index[:5]))
    return sets


def compute_shap_sets(df_train, df_test, cfg):
    """SHAP feature sets for load, DA price and ID price. Falls back to all features on error."""
    jobs = {
        "load": ("load", NON_FEATURES_LOAD),
        "price": ("dayahead_price", NON_FEATURES_PRICE),
        "id_price": ("intraday_price", NON_FEATURES_PRICE),
    }
    result = {}
    for name, (target, non_features) in jobs.items():
        try:
            result[name] = shap_rank_features(df_train, df_test, target, non_features, cfg)
        except Exception as e:
            log.warning("SHAP %s failed (%s), using all features", target, e)
            result[name] = {"all": [c for c in df_train.columns
                                    if c in df_test.columns and c not in non_features]}
    return result


def run(cfg):
    data_dir = cfg["data_dir"]
    df, fund_created, es_loaded, es_id_loaded = build_feature_table(data_dir)
    df = drop_all_nan_columns(df)
    df_train, df_test = get_train_test_split(df, cfg)
    log_summary(df, df_train, df_test, cfg)
    shap_sets = compute_shap_sets(df_train, df_test, cfg)
    return {
        "df": df,
        "df_train": df_train,
        "df_test": df_test,
        "fund_created": fund_created,
        "es_loaded": es_loaded,
        "es_id_loaded": es_id_loaded,
        "shap_sets_load": shap_sets["load"],
        "shap_sets_price": shap_sets["price"],
        "shap_sets_id_price": shap_sets["id_price"],
    }
