from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "merged_2018_2024_hourly_final.csv"


@pytest.fixture(scope="module")
def data():
    if not DATA.exists():
        pytest.skip("shipped dataset not present")
    return pd.read_csv(DATA, parse_dates=["datetime"])


def test_every_hour_present_once_in_utc(data):
    # local time: one missing hour each March, one doubled hour each October
    years = data["datetime"].dt.year.nunique()
    assert len(data) == len(pd.date_range("2018-01-01", "2024-12-31 23:00", freq="h"))
    assert data["datetime"].duplicated().sum() == years


def test_no_missing_values_outside_weather_lags(data):
    # era5 lags are empty for their first hours; the TSO load forecasts have occasional
    # publication gaps (HGB handles them, the TSO baseline fills them with the profile)
    skip = ("era5_", "load_fc_")
    core = data[[c for c in data.columns if not c.startswith(skip)]]
    assert not core.isna().any().any()
    # the forecast gaps stay small
    for c in ("load_fc_tso", "load_fc_es"):
        assert data[c].isna().mean() < 0.01


def test_load_is_consumption_not_supply(data):
    # consumption peaks below 10 GW; including exports and pumping pushed it to 12 GW with 3 GW jumps
    assert data["demand_mw"].max() < 10_500
    assert data["demand_mw"].diff().abs().max() < 1_500
    yearly = data.groupby(data["datetime"].dt.year)["demand_mw"].sum() / 1e6
    assert yearly.between(45, 55).all()


def test_daily_prices_constant_within_the_day(data):
    # a value that changes during the day would carry information from a later close
    per_day = data.groupby(data["datetime"].dt.date)[["ttf_eur_mwh", "carbon_eur_t"]].nunique()
    assert (per_day == 1).all().all()


def test_misaligned_radiation_feature_removed(data):
    assert not any("ssrd" in c for c in data.columns)


def test_plausible_ranges(data):
    assert data[[c for c in data.columns if c.startswith("gen_")]].min().min() >= 0
    assert data["congestion_ratio"].between(0, 2).all()
    assert np.isclose(data["da_price"].mean(), 81.5, atol=1.0)
