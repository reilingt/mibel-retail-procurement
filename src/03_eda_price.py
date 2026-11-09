"""Step 3: day-ahead vs intraday prices and their spread (by hour, month and weekday).
Takes df_train and df_test from step 1, saves tables as eda_price_*
and returns the hourly market rule.
"""
import logging

import numpy as np
import pandas as pd

import outputs

log = logging.getLogger(__name__)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def with_spread(df):
    out = df.copy()
    out["spread"] = out["intraday_price"] - out["dayahead_price"]
    return out


def build_hourly_market_rule(df_train):
    """Which market was cheaper on average, per hour and holiday flag."""
    rule = df_train.groupby(["hour", "is_holiday"]).agg(
        mean_DA=("dayahead_price", "mean"),
        mean_ID=("intraday_price", "mean"),
        std_DA=("dayahead_price", "std"),
        std_ID=("intraday_price", "std"),
        n=("dayahead_price", "size"),
    ).reset_index()
    rule["spread_id_da"] = rule["mean_ID"] - rule["mean_DA"]
    rule["cheaper_market"] = np.where(rule["mean_DA"] <= rule["mean_ID"], "DA", "ID")
    return rule


def build_hourly_spread_stats(df_train):
    sp = with_spread(df_train)
    rows = []
    for h in range(24):
        s = sp[sp["hour"] == h]["spread"]
        rows.append({
            "hour": h,
            "mean_spread": round(s.mean(), 2),
            "median_spread": round(s.median(), 2),
            "std_spread": round(s.std(), 2),
            "pct_id_cheaper": round((s < 0).mean() * 100, 1),
            "p10": round(s.quantile(0.10), 2),
            "p90": round(s.quantile(0.90), 2),
        })
    return pd.DataFrame(rows)


def build_monthly_prices(df_test):
    rows = []
    for m in range(1, 13):
        s = df_test[df_test["month"] == m]
        if s.empty:
            continue
        rows.append({
            "month": m,
            "mean_DA": round(s["dayahead_price"].mean(), 2),
            "mean_ID": round(s["intraday_price"].mean(), 2),
            "mean_spread": round((s["intraday_price"] - s["dayahead_price"]).mean(), 2),
            "std_DA": round(s["dayahead_price"].std(), 2),
            "max_DA": round(s["dayahead_price"].max(), 2),
            "min_DA": round(s["dayahead_price"].min(), 2),
        })
    return pd.DataFrame(rows)


def build_weekday_spread(df_test):
    sp = with_spread(df_test)
    rows = []
    for wd in range(7):
        s = sp[sp["weekday"] == wd]["spread"]
        rows.append({
            "weekday": WEEKDAYS[wd],
            "mean_spread": round(s.mean(), 2),
            "std_spread": round(s.std(), 2),
            "pct_id_cheaper": round((s < 0).mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def build_acf_tables(df_train, acf_lags=168):
    """Autocorrelation of DA, ID and spread up to one week, as arrays keyed DA/ID/spread."""
    out = {}
    try:
        from statsmodels.tsa.stattools import acf

        sp = with_spread(df_train)
        for col, label in [("dayahead_price", "DA"), ("intraday_price", "ID"), ("spread", "spread")]:
            s = sp[col].dropna()
            if len(s) > acf_lags:
                out[label] = acf(s, nlags=acf_lags)
    except ImportError:
        log.warning("statsmodels not installed, skipping ACF")
    except Exception as e:
        log.warning("ACF failed: %s", e)
    return out


def build_top50_spread(df_test):
    sp = with_spread(df_test)
    sp["abs_spread"] = sp["spread"].abs()
    top = sp.nlargest(50, "abs_spread")[["datetime", "dayahead_price", "intraday_price", "spread",
                                         "hour", "weekday", "is_holiday"]].copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    return top


def log_summary(tables):
    rule = tables["hourly_market_rule"]
    normal = rule[~rule["is_holiday"]][["hour", "mean_DA", "mean_ID", "spread_id_da", "cheaper_market"]]
    n_da = (normal["cheaper_market"] == "DA").sum()
    log.info("market rule (train, non-holiday):\n%s", normal.to_string(index=False))
    log.info("DA cheaper in %d/24 hours, ID cheaper in %d/24 hours", n_da, 24 - n_da)
    log.info("monthly prices (test):\n%s", tables["monthly_prices"].to_string(index=False))
    log.info("spread by weekday (test):\n%s", tables["weekday_spread"].to_string(index=False))
    hs = tables["hourly_spread"]
    log.info("hourly spread (train, top 5 by |spread|):\n%s",
             hs.reindex(hs["mean_spread"].abs().nlargest(5).index).to_string(index=False))


def run(cfg, df_train, df_test):
    rule = build_hourly_market_rule(df_train)
    tables = {
        "hourly_market_rule": rule,
        "hourly_spread": build_hourly_spread_stats(df_train),
        "monthly_prices": build_monthly_prices(df_test),
        "weekday_spread": build_weekday_spread(df_test),
    }
    tables.update(build_acf_tables(df_train))
    tables["top50_spread"] = build_top50_spread(df_test)
    market_rule = dict(zip(zip(rule["hour"], rule["is_holiday"]), rule["cheaper_market"]))
    log_summary(tables)
    acf = {k: tables[k] for k in ("DA", "ID", "spread") if k in tables}
    saved = {**tables, "acf": pd.DataFrame(acf)} if acf else tables
    outputs.save(saved, cfg["out_dir"], "eda_price")
    return {"tables": tables, "market_rule": market_rule}
