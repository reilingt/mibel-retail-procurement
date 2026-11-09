"""Step 6: procurement cost without a battery for every load/price pair, plus the naive
baselines and the perfect-foresight oracle. Writes procurement_results_{scenario}.csv and _all.csv."""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LOAD_LABELS = {
    "fc_profile": "Profile", "fc_arx_da": "ARX", "fc_tso": "TSO",
    "fc_lear_da": "LEAR_Lasso", "fc_lear_ridge_da": "LEAR_Ridge", "fc_lear_enet_da": "LEAR_ENet",
}


def load_label(col):
    """'fc_hgb_static_1y_shap_p90' -> 'HGB__static_1y__shap_p90'."""
    if col in LOAD_LABELS:
        return LOAD_LABELS[col]
    m = re.match(r"fc_(hgb|lstm|dnn)_(.+)_shap_(.+)$", col)
    return f"{m.group(1).upper()}__{m.group(2)}__shap_{m.group(3)}" if m else col.replace("fc_", "")


def da_to_id(col):
    """fc_hgb_da_rolling_6m_daily_shap_p90 -> fc_hgb_id_... (the lookahead keeps _daily intact)."""
    return re.sub(r"_da(?=_|$)", "_id", col)


def surplus_price(price_id, mode, factor):
    if mode == "id_sell":
        return price_id
    if mode == "imbalance":
        return price_id * factor
    return 0.0


def hourly_cost(load_fc, price_fc, fc_col, market, share, mode, factor):
    """Hourly cost of buying share * forecast in the chosen market; deviations settle at the ID price."""
    load_actual = load_fc["load_actual"].values
    fc_load = load_fc[fc_col].fillna(load_fc["fc_profile"]).values
    price_da = price_fc["dayahead_price"].values
    price_id = price_fc["intraday_price"].values

    buy_qty = share * fc_load
    buy_price = np.where(market == "DA", price_da, price_id)
    deviation = share * load_actual - buy_qty
    shortage = np.maximum(0, deviation)
    surplus = np.maximum(0, -deviation)
    cost_shortage = shortage * price_id
    cost_surplus = -surplus * surplus_price(price_id, mode, factor)
    cost_total = buy_qty * buy_price + cost_shortage + cost_surplus
    return cost_total, cost_shortage, cost_surplus, shortage, surplus


def simulate(load_fc, price_fc, fc_col, market, share, mode, factor):
    cost_total, cost_shortage, cost_surplus, shortage, surplus = hourly_cost(
        load_fc, price_fc, fc_col, market, share, mode, factor)
    return {
        "total_cost_eur": round(pd.Series(cost_total).sum(), 2),
        "imbalance_cost_eur": round(pd.Series(cost_shortage).sum(), 2),
        "surplus_revenue_eur": round(pd.Series(cost_surplus).sum(), 2),
        "surplus_mwh": round(pd.Series(surplus).sum(), 2),
        "shortage_mwh": round(pd.Series(shortage).sum(), 2),
    }


def price_pairs(price_fc):
    """DA forecast columns that have an ID column with the same model and config."""
    da_cols = [c for c in price_fc.columns
               if c.startswith("fc_") and ("_da_" in c or c.endswith("_da")) and price_fc[c].notna().sum() > 100]
    id_cols = {c for c in price_fc.columns
               if c.startswith("fc_") and ("_id_" in c or c.endswith("_id")) and price_fc[c].notna().sum() > 100}
    pairs = [(c, da_to_id(c)) for c in da_cols if da_to_id(c) in id_cols]
    unpaired = [c for c in da_cols if c not in {d for d, _ in pairs}]
    if unpaired:
        log.info("DA columns without ID partner: %s", unpaired)
    return pairs


def static_market(price_fc, load_fc, df_train):
    """market_static from step 5 (per hour and holiday). If missing, an hour-only rule from df_train."""
    if "market_static" in price_fc.columns:
        return price_fc["market_static"].values
    hourly = df_train.groupby("hour").agg(avg_DA=("dayahead_price", "mean"), avg_ID=("intraday_price", "mean"))
    rule = dict(zip(hourly.index, np.where(hourly["avg_DA"] <= hourly["avg_ID"], "DA", "ID")))
    return load_fc["hour"].map(rule).values


def run_scenario(load_fc, price_fc, df_train, share, mode, factor):
    load_cols = [c for c in load_fc.columns if c.startswith("fc_") and c != "fc_load"
                 and load_fc[c].notna().sum() > 100]
    pairs = price_pairs(price_fc)
    log.info("%d load forecasts x %d price pairs", len(load_cols), len(pairs))

    oracle_price = np.minimum(price_fc["dayahead_price"].values, price_fc["intraday_price"].values)
    rows = [{
        "method": "Oracle_perfect", "load_model": "oracle", "price_model": "oracle", "market_method": "oracle",
        "total_cost_eur": round(pd.Series(share * load_fc["load_actual"].values * oracle_price).sum(), 2),
        "imbalance_cost_eur": 0.0, "surplus_revenue_eur": 0.0, "surplus_mwh": 0.0, "shortage_mwh": 0.0,
    }]

    # naive baselines: profile load forecast, everything in one market
    n = len(load_fc)
    for mkt in ("DA", "ID"):
        rows.append({"method": f"Naive_profile_all_{mkt}", "load_model": "profile",
                     "price_model": f"naive_all_{mkt.lower()}", "market_method": "naive",
                     **simulate(load_fc, price_fc, "fc_profile", np.full(n, mkt), share, mode, factor)})

    # every load forecast with everything bought day-ahead, to separate load and routing value
    for col in load_cols:
        rows.append({"method": f"{load_label(col)}__all_DA", "load_model": col.replace("fc_", ""),
                     "price_model": "all_da", "market_method": "all_da",
                     **simulate(load_fc, price_fc, col, np.full(n, "DA"), share, mode, factor)})

    market = static_market(price_fc, load_fc, df_train)
    for col in load_cols:
        rows.append({"method": f"{load_label(col)}__static", "load_model": col.replace("fc_", ""),
                     "price_model": "static", "market_method": "static",
                     **simulate(load_fc, price_fc, col, market, share, mode, factor)})

    for da_col, id_col in pairs:
        market = np.where(price_fc[id_col].values < price_fc[da_col].values, "ID", "DA")
        for col in load_cols:
            rows.append({"method": f"{load_label(col)}__{da_col}", "load_model": col.replace("fc_", ""),
                         "price_model": da_col, "market_method": "forecast",
                         **simulate(load_fc, price_fc, col, market, share, mode, factor)})

    total_mwh = share * load_fc["load_actual"].sum()
    results = pd.DataFrame(rows).sort_values("total_cost_eur").reset_index(drop=True)
    results["cost_per_mwh"] = (results["total_cost_eur"] / total_mwh).round(2)
    # The imbalance part only depends on the load forecast, not on the routing
    results["imbalance_per_mwh"] = (results["imbalance_cost_eur"] / total_mwh).round(4)
    return results


def common_eval_period(load_fc, price_fc, cfg):
    """Keep only hours where every forecast has a value (drops the warm-up of rolling models)."""
    start = cfg["eval_start_date"]
    if start:
        mask = pd.to_datetime(load_fc["datetime"]) >= pd.Timestamp(start)
    else:
        load_cols = [c for c in load_fc.columns if c.startswith("fc_") and load_fc[c].notna().sum() > 100]
        price_cols = [c for c in price_fc.columns if c.startswith("fc_") and price_fc[c].notna().sum() > 100]
        mask = load_fc[load_cols].notna().all(axis=1).values & price_fc[price_cols].notna().all(axis=1).values
    log.info("evaluation period: %d of %d hours", mask.sum(), len(mask))
    return load_fc[mask].reset_index(drop=True), price_fc[mask].reset_index(drop=True)


def log_summary(results, label):
    no_oracle = results[results["method"] != "Oracle_perfect"]
    oracle = results.loc[results["method"] == "Oracle_perfect", "cost_per_mwh"]
    static = no_oracle[no_oracle["price_model"] == "static"].sort_values("imbalance_per_mwh")
    naive = no_oracle[no_oracle["market_method"] == "naive"]
    log.info("[%s] naive baselines:\n%s", label, naive[["method", "cost_per_mwh"]].to_string(index=False))
    top = no_oracle[no_oracle["market_method"] == "forecast"].sort_values("cost_per_mwh").head(20)
    log.info("[%s] oracle: %.2f EUR/MWh", label, oracle.iloc[0] if len(oracle) else np.nan)
    log.info("[%s] load models (static routing, by imbalance):\n%s", label,
             static[["load_model", "imbalance_per_mwh", "cost_per_mwh"]].to_string(index=False))
    log.info("[%s] top 20 load x price pairs:\n%s", label,
             top[["load_model", "price_model", "cost_per_mwh"]].to_string(index=False))


def run(cfg, load_fc, price_fc, df_train):
    out_dir = Path(cfg["out_dir"])
    load_fc, price_fc = common_eval_period(load_fc, price_fc, cfg)

    results = {}
    for sc in cfg["surplus_scenarios"]:
        res = run_scenario(load_fc, price_fc, df_train, cfg["share"],
                           sc["surplus_mode"], sc["imbalance_price_factor"])
        res["scenario"] = sc["label"]
        results[sc["label"]] = res
        log_summary(res, sc["label"])
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            res.to_csv(out_dir / f"procurement_results_{sc['label']}.csv", index=False)
        except Exception as e:
            log.warning("saving %s failed: %s", sc["label"], e)

    combined = pd.concat(results.values(), ignore_index=True)
    try:
        combined.to_csv(out_dir / "procurement_results_all.csv", index=False)
    except Exception as e:
        log.warning("saving combined results failed: %s", e)

    if "zero" in results:
        ref = results["zero"].loc[lambda r: r["method"] != "Oracle_perfect"].sort_values("total_cost_eur").iloc[0]
        for label, res in results.items():
            best = res[res["method"] != "Oracle_perfect"].sort_values("total_cost_eur").iloc[0]
            log.info("%-6s best %.2f EUR/MWh (%+.0f EUR vs zero), surplus %.1f MWh", label, best["cost_per_mwh"],
                     best["total_cost_eur"] - ref["total_cost_eur"], best["surplus_mwh"])

    return {"procurement_results": results, "procurement_results_all": combined}
