import numpy as np
import pandas as pd
import pytest

from conftest import step

proc = step("06_procurement")


@pytest.fixture
def frames():
    # step 6 skips forecast columns with 100 or fewer values, so use one week
    n = 24 * 7
    hours = np.arange(n) % 24
    da = np.where(hours < 12, 50.0, 80.0)
    idp = np.where(hours < 12, 60.0, 70.0)
    load = np.full(n, 1000.0)
    load_fc = pd.DataFrame({"hour": hours, "load_actual": load, "fc_profile": load * 0.9,
                            "fc_perfect": load})
    price_fc = pd.DataFrame({"dayahead_price": da, "intraday_price": idp,
                             "fc_good_da": da, "fc_good_id": idp,
                             "market_static": np.where(hours < 12, "DA", "ID")})
    return load_fc, price_fc


def test_perfect_foresight_equals_oracle(frames):
    load_fc, price_fc = frames
    market = np.where(price_fc["intraday_price"] < price_fc["dayahead_price"], "ID", "DA")
    res = proc.simulate(load_fc, price_fc, "fc_perfect", market, 0.01, "zero", 0.0)
    oracle = (0.01 * load_fc["load_actual"] * np.minimum(price_fc["dayahead_price"],
                                                         price_fc["intraday_price"])).sum()
    assert res["total_cost_eur"] == pytest.approx(oracle)
    assert res["shortage_mwh"] == 0 and res["surplus_mwh"] == 0


def test_shortage_is_settled_at_intraday_price(frames):
    load_fc, price_fc = frames
    market = np.full(len(load_fc), "DA")
    res = proc.simulate(load_fc, price_fc, "fc_profile", market, 0.01, "zero", 0.0)
    expected = (0.01 * 0.1 * load_fc["load_actual"] * price_fc["intraday_price"]).sum()
    assert res["imbalance_cost_eur"] == pytest.approx(expected)


def test_surplus_revenue_scales_with_factor(frames):
    load_fc, price_fc = frames
    load_fc = load_fc.assign(fc_over=load_fc["load_actual"] * 1.1)
    market = np.full(len(load_fc), "DA")
    zero = proc.simulate(load_fc, price_fc, "fc_over", market, 0.01, "zero", 0.0)
    sur90 = proc.simulate(load_fc, price_fc, "fc_over", market, 0.01, "imbalance", 0.9)
    revenue = (0.01 * 0.1 * load_fc["load_actual"] * price_fc["intraday_price"] * 0.9).sum()
    assert zero["total_cost_eur"] - sur90["total_cost_eur"] == pytest.approx(revenue)


def test_scenario_has_oracle_naive_and_forecast_rows(frames):
    load_fc, price_fc = frames
    res = proc.run_scenario(load_fc, price_fc, None, 0.01, "zero", 0.0)
    methods = set(res["market_method"])
    assert {"oracle", "naive", "static", "forecast"} <= methods
    assert res.loc[res["market_method"] == "oracle", "total_cost_eur"].iloc[0] == res["total_cost_eur"].min()


sig = step("08_significance")


def test_bootstrap_interval_excludes_zero_for_clear_effect():
    x = np.random.default_rng(0).normal(1.0, 0.5, 365)
    boot = sig.block_bootstrap_mean(x, n_boot=2000)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert lo > 0 and lo < x.mean() < hi


def test_diebold_mariano_identical_forecasts():
    err = np.random.default_rng(0).normal(size=24 * 30)
    days = np.repeat(np.arange(30), 24)
    stat, p, n = sig.diebold_mariano(err, err, days)
    assert p == 1.0 and n == 30
