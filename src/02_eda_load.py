"""Step 2: descriptive analysis of Portuguese load in the test year (monthly pattern,
top and bottom days, peak hours, holiday and bridge-day effects).
Takes df, df_train and df_test from step 1, saves the tables as eda_load_*.
"""
import logging
from datetime import timedelta

import holidays
import numpy as np
import pandas as pd

import outputs

log = logging.getLogger(__name__)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_monthly_patterns(df_year):
    rows = []
    for m in range(1, 13):
        s = df_year[df_year["month"] == m].dropna(subset=["load"])
        if s.empty:
            continue
        daily = s.groupby("date")["load"].mean()
        if daily.empty or not np.isfinite(daily).any():
            continue
        peak_date = daily.idxmax()
        peak_row = s[s["date"] == peak_date].nlargest(1, "load").iloc[0]
        rows.append({
            "month": m,
            "mean_load_mw": round(s["load"].mean(), 1),
            "max_load_mw": round(s["load"].max(), 1),
            "min_load_mw": round(s["load"].min(), 1),
            "day_of_peak": str(peak_date),
            "hour_of_peak": int(peak_row["hour"]),
            "std_mw": round(s["load"].std(), 1),
            "cv": round(s["load"].std() / s["load"].mean(), 3),
        })
    return pd.DataFrame(rows)


def build_day_table(df_year, ascending, n=3):
    """The n days with the highest (ascending=False) or lowest mean load per month."""
    rows = []
    for m in range(1, 13):
        s = df_year[df_year["month"] == m].dropna(subset=["load"])
        if s.empty:
            continue
        daily = s.groupby("date").agg(mean_load=("load", "mean"), peak_load=("load", "max"))
        if daily.empty:
            continue
        valid_dates = daily.index[daily["peak_load"].notna()]
        if len(valid_dates) == 0:
            continue
        s_valid = s[s["date"].isin(valid_dates)]
        peak_hours = s_valid.loc[s_valid.groupby("date")["load"].idxmax()].set_index("date")["hour"].rename("peak_hour")
        daily = daily.join(peak_hours).sort_values("mean_load", ascending=ascending).head(n)
        for rank, (d, r) in enumerate(daily.iterrows(), 1):
            info = s[s["date"] == d].iloc[0]
            rows.append({
                "month": m, "rank": rank, "date": str(d),
                "weekday": pd.Timestamp(d).day_name(),
                "mean_load_mw": round(r["mean_load"], 1),
                "peak_hour": int(r["peak_hour"]),
                "peak_load_mw": round(r["peak_load"], 1),
                "is_holiday": bool(info["is_holiday"]),
                "holiday_name": info["holiday_name"],
            })
    return pd.DataFrame(rows)


def build_peak_hours(df_year):
    """Mean load per hour and the share of days on which that hour was the daily peak."""
    valid = df_year.dropna(subset=["load"])
    peak_hours = pd.Series(dtype=float)
    if len(valid):
        peak_hours = valid.loc[valid.groupby("date")["load"].idxmax(), "hour"]
    n_days = df_year["date"].nunique()
    rows = []
    for h in range(24):
        sh = valid[valid["hour"] == h]
        rows.append({
            "hour": h,
            "mean_load_mw": round(sh["load"].mean(), 1),
            "frac_peak": round((peak_hours == h).sum() / n_days, 4) if n_days > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def build_holidays_table(df, df_train, test_year, pt_holidays):
    """Load on each holiday vs. the mean of the same weekday in the training window."""
    years = [test_year - 2, test_year - 1, test_year]
    dates_by_name = {}
    for d, name in sorted(pt_holidays.items()):
        if d.year in years:
            dates_by_name.setdefault(name, {})[d.year] = d

    rows = []
    for name in sorted(dates_by_name):
        row = {"holiday_name": name}
        for y in years:
            d = dates_by_name[name].get(y)
            row[f"date_{y}"] = str(d) if d else ""
            if d is None:
                row[f"mean_load_{y}"] = np.nan
                row[f"pct_dev_{y}"] = np.nan
                continue
            day = df[df["date"] == d]
            holiday_mean = day["load"].mean() if len(day) else np.nan
            row[f"mean_load_{y}"] = round(holiday_mean, 1) if pd.notna(holiday_mean) else np.nan
            normal = df_train[(df_train["weekday"] == d.weekday()) & (~df_train["is_holiday"])]["load"].mean()
            row[f"pct_dev_{y}"] = (round((holiday_mean - normal) / normal * 100, 1)
                                   if normal > 0 and pd.notna(holiday_mean) else np.nan)
        hours = df[df["holiday_name"] == name]
        row["mean_DA_price"] = round(hours["dayahead_price"].mean(), 2)
        row["mean_spread"] = round((hours["intraday_price"] - hours["dayahead_price"]).mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def build_bridge_days_table(df, df_train, pt_holidays):
    rows = []
    for d in sorted(df[df["is_bridge_day"]]["date"].unique()):
        day = df[df["date"] == d]
        wd = pd.Timestamp(d).weekday()
        adjacent = (d - timedelta(days=1)) if wd == 4 else ((d + timedelta(days=1)) if wd == 0 else None)
        normal_days = (df_train["weekday"] == wd) & ~df_train["is_holiday"] & ~df_train["is_bridge_day"]
        normal = df_train[normal_days]["load"].mean()
        rows.append({
            "date": str(d),
            "adjacent_holiday": pt_holidays.get(adjacent, "") if adjacent else "",
            "weekday": pd.Timestamp(d).day_name(),
            "mean_load_mw": round(day["load"].mean(), 1),
            "pct_dev": round((day["load"].mean() - normal) / normal * 100, 1),
            "mean_DA": round(day["dayahead_price"].mean(), 2),
        })
    return pd.DataFrame(rows)


def build_top50_hours(df_year):
    top = df_year.dropna(subset=["load"]).nlargest(50, "load").copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top = top[["rank", "datetime", "load", "hour", "weekday", "is_holiday", "month"]]
    return top.rename(columns={"load": "load_mw"})


def build_weekday_month_table(df_year):
    pivot = df_year.pivot_table(values="load", index="weekday", columns="month", aggfunc="mean")
    if len(pivot.index) == 7:
        pivot.index = WEEKDAYS
    pivot.columns = [f"M{m:02d}" for m in pivot.columns]
    return pivot.round(0)


def log_summary(tables, df_year, test_year):
    log.info("monthly patterns %d:\n%s", test_year, tables["monthly_patterns"].to_string(index=False))
    log.info("peak hours (top 5):\n%s", tables["peak_hours"].nlargest(5, "frac_peak").to_string(index=False))
    cols = ["holiday_name"] + [f"pct_dev_{y}" for y in [test_year - 2, test_year - 1, test_year]]
    log.info("holiday load impact:\n%s",
             tables["holidays"][[c for c in cols if c in tables["holidays"].columns]].to_string(index=False))
    log.info("weekday x month (mean MW):\n%s", tables["weekday_month"].to_string())

    monthly = tables["monthly_patterns"]
    if len(monthly):
        hi = monthly.loc[monthly["mean_load_mw"].idxmax()]
        lo = monthly.loc[monthly["mean_load_mw"].idxmin()]
        log.info("highest month: %d (%.0f MW), lowest month: %d (%.0f MW)",
                 hi["month"], hi["mean_load_mw"], lo["month"], lo["mean_load_mw"])

    daily = df_year.dropna(subset=["load"]).groupby("date")["load"].mean()
    if len(daily):
        log.info("peak day: %s (%.0f MW), low day: %s (%.0f MW)",
                 daily.idxmax(), daily.max(), daily.idxmin(), daily.min())

    peaks = tables["peak_hours"]
    if len(peaks) and peaks["frac_peak"].notna().any():
        top = peaks.loc[peaks["frac_peak"].idxmax()]
        log.info("most common peak hour: %d:00 (%.1f%% of days)", top["hour"], top["frac_peak"] * 100)


def run(cfg, df, df_train, df_test):
    test_year = cfg["test_year"]
    pt_holidays = holidays.country_holidays("PT", years=[test_year - 2, test_year - 1, test_year])
    tables = {
        "monthly_patterns": build_monthly_patterns(df_test),
        "top_days": build_day_table(df_test, ascending=False),
        "bottom_days": build_day_table(df_test, ascending=True),
        "peak_hours": build_peak_hours(df_test),
        "holidays": build_holidays_table(df, df_train, test_year, pt_holidays),
        "bridge_days": build_bridge_days_table(df, df_train, pt_holidays),
        "top50_hours": build_top50_hours(df_test),
        "weekday_month": build_weekday_month_table(df_test),
    }
    log_summary(tables, df_test, test_year)
    outputs.save(tables, cfg["out_dir"], "eda_load")
    return {"tables": tables}
