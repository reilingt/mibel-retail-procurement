import numpy as np
import pandas as pd

from conftest import step

prep = step("01_data_prep")

# published the day before, so the same-hour value is a legitimate input
DAY_AHEAD_KNOWN = {"wind_fc", "solar_fc", "load_fc_tso", "load_fc_es"}


def build_features(df):
    df = df.copy()
    df, _ = prep.add_lag_features(df)
    df, _ = prep.add_spanish_features(df, "da_price_es", "dayahead_price", "spread_es_pt")
    df, _ = prep.add_spanish_features(df, "id_price_es", "intraday_price", "spread_id_es_pt")
    df = prep.add_rolling_features(df)
    return prep.add_spread_features(df)


def test_features_only_use_data_known_at_gate_closure(hourly_frame):
    # change everything from t-23 onwards; no model input at t may move
    t = len(hourly_frame) - 30
    base = build_features(hourly_frame)

    shocked = hourly_frame.copy()
    raw = [c for c in shocked.columns if c != "datetime" and c not in DAY_AHEAD_KNOWN]
    shocked.loc[t - 23:, raw] += 1000.0
    after = build_features(shocked)

    inputs = [c for c in base.columns if c not in prep.NON_FEATURES and c not in DAY_AHEAD_KNOWN]
    assert inputs, "no feature columns built"
    changed = [c for c in inputs if not np.isclose(base.at[t, c], after.at[t, c], equal_nan=True)]
    assert not changed, f"features leak information from the last 24 h: {changed}"


def test_no_lag_shorter_than_one_day():
    assert min(prep.LAGS) >= 24


def test_dst_duplicate_timestamp_is_dropped(tmp_path):
    ts = ["2024-10-27 00:00", "2024-10-27 01:00", "2024-10-27 01:00", "2024-10-27 02:00"]
    pd.DataFrame({"datetime": ts, "da_price": [1, 2, 3, 4]}).to_csv(tmp_path / prep.RAW_CSV, index=False)
    df = prep.load_data(tmp_path)
    assert df["datetime"].is_unique
    assert len(df) == 3


def test_rolling_train_window_ends_before_target_day(hourly_frame):
    target = hourly_frame["datetime"].iloc[24 * 15].normalize()
    mask = prep.get_train_mask(hourly_frame, target, "rolling_6m", 2023)
    assert hourly_frame.loc[mask, "datetime"].max() < target


def test_shap_selection_never_looks_at_test_year(hourly_frame):
    df = build_features(hourly_frame).dropna().reset_index(drop=True)
    train, test = df.iloc[:300].copy(), df.iloc[300:].copy()
    cfg = {"hgb_max_iter": 20, "hgb_max_depth": 3, "hgb_learning_rate": 0.1,
           "hgb_min_samples_leaf": 10, "hgb_random_state": 0, "shap_thresholds": [0.9, 1.0]}

    sets_a = prep.shap_rank_features(train, test, "dayahead_price", prep.NON_FEATURES_PRICE, cfg)
    noisy = test.copy()
    num = noisy.select_dtypes("number").columns
    noisy[num] = np.random.default_rng(1).normal(size=(len(noisy), len(num)))
    sets_b = prep.shap_rank_features(train, noisy, "dayahead_price", prep.NON_FEATURES_PRICE, cfg)
    assert sets_a == sets_b
