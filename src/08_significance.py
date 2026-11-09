"""Step 8: significance of the headline gaps. Cost via a moving-block bootstrap over days,
price accuracy via Diebold-Mariano. Writes significance.csv and routing_accuracy.csv."""
import importlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import SEED

log = logging.getLogger(__name__)
proc = importlib.import_module("06_procurement")

N_BOOT = 10_000
BLOCK_DAYS = 7


def block_bootstrap_mean(x, n_boot=N_BOOT, block=BLOCK_DAYS, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return x[idx].mean(axis=1)


def newey_west_var(d, lags):
    d = d - d.mean()
    n = len(d)
    var = d @ d / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return var / n


def diebold_mariano(err_a, err_b, days, lags=BLOCK_DAYS - 1):
    # loss = absolute error, averaged per day; positive stat means a is worse
    loss = pd.Series(np.abs(err_a) - np.abs(err_b)).groupby(days).mean().values
    var = newey_west_var(loss, lags)
    if var <= 0:
        return 0.0, 1.0, len(loss)
    stat = loss.mean() / np.sqrt(var)
    return stat, 2 * stats.norm.sf(abs(stat)), len(loss)


def compare_costs(cost_a, cost_b, days, mwh_per_day):
    # positive diff means a is more expensive than b
    daily = (pd.Series(cost_a - cost_b).groupby(days).sum() / mwh_per_day).values
    boot = block_bootstrap_mean(daily)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return daily.mean(), lo, hi, p, len(daily)


def routing(price_fc, da_col):
    if da_col == "static":
        return price_fc["market_static"].values
    if da_col.startswith("naive_all_"):
        return np.full(len(price_fc), da_col[-2:].upper())
    id_col = proc.da_to_id(da_col)
    return np.where(price_fc[id_col].values < price_fc[da_col].values, "ID", "DA")


def run(cfg, load_fc, price_fc, df_train, results=None):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    load_fc, price_fc = proc.common_eval_period(load_fc, price_fc, cfg)
    if "market_static" not in price_fc.columns:
        price_fc = price_fc.assign(market_static=proc.static_market(price_fc, load_fc, df_train))
    days = pd.to_datetime(load_fc["datetime"]).dt.date.values
    share = cfg["share"]
    mwh_per_day = (pd.Series(share * load_fc["load_actual"].values).groupby(days).sum())
    mwh_per_day = mwh_per_day.values

    rows = []
    for sc in cfg["surplus_scenarios"]:
        res = proc.run_scenario(load_fc, price_fc, df_train, share, sc["surplus_mode"], sc["imbalance_price_factor"])
        fc = res[res["market_method"] == "forecast"].sort_values("total_cost_eur")
        best = fc.iloc[0]
        load_col = "fc_" + best["load_model"]
        same_load = fc[fc["load_model"] == best["load_model"]]
        challengers = {
            "best other price model": same_load[same_load["price_model"] != best["price_model"]]["price_model"].iloc[0],
            "static hourly rule": "static",
            "naive, all day-ahead": "naive_all_da",
        }

        def cost(load, price_model):
            col = "fc_profile" if price_model.startswith("naive_all_") else load
            return proc.hourly_cost(load_fc, price_fc, col, routing(price_fc, price_model), share,
                                    sc["surplus_mode"], sc["imbalance_price_factor"])[0]

        base = cost(load_col, best["price_model"])
        for label, other in challengers.items():
            diff, lo, hi, p, n = compare_costs(cost(load_col, other), base, days, mwh_per_day)
            rows.append({"scenario": sc["label"], "test": "cost block bootstrap",
                         "a": other, "b": best["price_model"], "load_model": best["load_model"],
                         "comparison": f"{label} minus best", "diff": round(diff, 3),
                         "ci_low": round(lo, 3), "ci_high": round(hi, 3), "p_value": round(p, 4),
                         "n_days": n, "unit": "EUR/MWh"})

    # price accuracy: ARX vs the most accurate other DA model
    actual = price_fc["dayahead_price"].values
    da_cols = [c for c in price_fc.columns if c.startswith("fc_") and re.search(r"_da(_|$)", c)
               and price_fc[c].notna().sum() > 100]
    mae = {c: np.nanmean(np.abs(price_fc[c].values - actual)) for c in da_cols}
    if "fc_arx_da" in mae and len(mae) > 1:
        rival = min((c for c in mae if c != "fc_arx_da"), key=mae.get)
        stat, p, n = diebold_mariano(price_fc["fc_arx_da"].values - actual, price_fc[rival].values - actual, days)
        rows.append({"scenario": "all", "test": "Diebold-Mariano (abs error)", "a": "fc_arx_da", "b": rival,
                     "load_model": "", "comparison": "ARX MAE minus rival MAE",
                     "diff": round(mae["fc_arx_da"] - mae[rival], 3), "ci_low": np.nan, "ci_high": np.nan,
                     "p_value": round(p, 4), "n_days": n, "unit": f"EUR/MWh (DM stat {stat:.2f})"})

    # routing accuracy: share of hours where the chosen market was the cheaper one
    cheaper = np.where(price_fc["intraday_price"].values < price_fc["dayahead_price"].values, "ID", "DA")
    static = routing(price_fc, "static")
    models = ["static"] + [c for c in da_cols if proc.da_to_id(c) in price_fc.columns]
    routing_acc = pd.DataFrame({
        "model": models,
        "routing_acc_pct": [round((routing(price_fc, m) == cheaper).mean() * 100, 1) for m in models],
        "agrees_with_static_pct": [round((routing(price_fc, m) == static).mean() * 100, 1) for m in models],
    }).sort_values("routing_acc_pct", ascending=False)
    routing_acc.to_csv(out_dir / "routing_accuracy.csv", index=False)
    log.info("routing accuracy:\n%s", routing_acc.to_string(index=False))

    table = pd.DataFrame(rows)
    log.info("significance:\n%s", table.to_string(index=False))
    table.to_csv(out_dir / "significance.csv", index=False)
    return {"significance": table}
