"""Step 7: battery dispatch on top of procurement, three strategies across several sizes.
Writes bess_results_{scenario}.csv and bess_results_all.csv."""
import logging
from math import floor, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def bess_common(cfg):
    return {
        "eta": cfg["bess_eta"],
        "soc_min": cfg["bess_soc_min"],
        "soc_max": cfg["bess_soc_max"],
        "cycles_per_day": cfg["bess_cycles_per_day"],
        "capex_eur_per_kwh": cfg["bess_capex_eur_per_kwh"],
        "lifetime_years": cfg["bess_lifetime_years"],
        "discount_rate": cfg["bess_discount_rate"],
        "om_pct_capex": cfg["bess_om_pct_capex"],
        "grid_fee_eur_mwh": cfg["bess_grid_fee_eur_mwh"],
    }


class Settlement:
    """Hourly cost: bought energy at the actual price, shortage at ID, surplus by scenario.
    The battery's charge/discharge is part of the retailer's energy balance."""

    def __init__(self, surplus_mode, imbalance_factor):
        self.mode = surplus_mode
        self.factor = imbalance_factor

    def surplus_price(self, price_id):
        if self.mode == "id_sell":
            return price_id
        if self.mode == "imbalance":
            return price_id * self.factor
        return 0.0

    def __call__(self, grid_buy, buy_price, retailer_load, price_id, chg, dis):
        load = retailer_load + chg - dis
        deviation = load - grid_buy
        short = max(0, deviation)
        sur = max(0, -deviation)
        cost = grid_buy * buy_price + short * price_id - sur * self.surplus_price(price_id)
        return cost, short * price_id, short, sur


def battery(params):
    E = params["E_max_MWh"]
    return params["P_max_MW"], E, sqrt(params["eta"]), params["soc_min"] * E, params["soc_max"] * E


def run_arbitrage(data, params, settle):
    """Charge in the K cheapest hours of each day, discharge in the K most expensive."""
    P, E, eta_s, soc_lo, soc_hi = battery(params)
    k = max(1, floor(params["cycles_per_day"] * E / P))
    soc = 0.5 * (soc_lo + soc_hi)
    tc = ti = tsh = tsur = tst = 0.0

    for _, day in data.groupby("date"):
        day = day.sort_values("hour")
        prices = day.set_index("hour")["buy_price"]
        charge_hours = set(prices.nsmallest(k).index)
        discharge_hours = set(prices.nlargest(k).index)

        for _, r in day.iterrows():
            h = int(r["hour"])
            chg, dis = 0.0, 0.0
            if r["buy_price"] < 0 and soc < soc_hi:  # paid to consume: always charge
                chg = min(P, max(0, (soc_hi - soc) / eta_s))
            elif h in charge_hours:
                chg = min(P, max(0, (soc_hi - soc) / eta_s))
            elif h in discharge_hours:
                dis = min(P, max(0, (soc - soc_lo) * eta_s))
            soc = np.clip(soc + eta_s * chg - dis / eta_s, soc_lo, soc_hi)

            grid = max(0, r["buy_forecast"] + chg - dis)
            cost, imb, short, sur = settle(grid, r["buy_price"], r["retailer_load"], r["price_id"], chg, dis)
            tc += cost
            ti += imb
            tsh += short
            tsur += sur
            tst += chg
    return tc, ti, tsur, tsh, tst


def run_balancing(data, params, settle):
    """Use the battery to cover the load forecast error hour by hour."""
    P, E, eta_s, soc_lo, soc_hi = battery(params)
    soc = 0.5 * (soc_lo + soc_hi)
    tc = ti = tsh = tsur = tst = 0.0

    for _, r in data.iterrows():
        dev = r["retailer_load"] - r["buy_forecast"]
        chg, dis = 0.0, 0.0
        if r["buy_price"] < 0 and soc < soc_hi:
            chg = min(P, max(0, (soc_hi - soc) / eta_s))
        elif dev < 0:  # bought too much -> charge
            chg = min(P, -dev, max(0, (soc_hi - soc) / eta_s))
        elif dev > 0:  # bought too little -> discharge
            dis = min(P, dev, max(0, (soc - soc_lo) * eta_s))

        soc = np.clip(soc + eta_s * chg - dis / eta_s, soc_lo, soc_hi)
        grid = max(0, r["buy_forecast"] + chg - dis)
        cost, imb, short, sur = settle(grid, r["buy_price"], r["retailer_load"], r["price_id"], chg, dis)
        tc += cost
        ti += imb
        tsh += short
        tsur += sur
        tst += chg
    return tc, ti, tsur, tsh, tst


def run_spread(data, params, settle, threshold):
    """Charge if the forecast DA-ID spread is above the threshold, discharge if below, balance with the rest."""
    P, E, eta_s, soc_lo, soc_hi = battery(params)
    soc = 0.5 * (soc_lo + soc_hi)
    tc = ti = tsh = tsur = tst = 0.0

    for _, r in data.iterrows():
        spread = r["spread_fc"]
        chg, dis = 0.0, 0.0
        if r["buy_price"] < 0 and soc < soc_hi:
            chg = min(P, max(0, (soc_hi - soc) / eta_s))
        elif spread > threshold and soc < soc_hi:
            chg = min(P, max(0, (soc_hi - soc) / eta_s))
        elif spread < -threshold and soc > soc_lo:
            dis = min(P, max(0, (soc - soc_lo) * eta_s))
        soc = np.clip(soc + eta_s * chg - dis / eta_s, soc_lo, soc_hi)

        grid = max(0, r["buy_forecast"] + chg - dis)
        deviation = r["retailer_load"] - (grid + dis - chg)
        p_left = P - max(chg, dis)
        extra_chg, extra_dis = 0.0, 0.0
        if p_left > 0:
            if deviation > 0 and soc > soc_lo:
                extra_dis = min(p_left, deviation, max(0, (soc - soc_lo) * eta_s))
                soc -= extra_dis / eta_s
            elif deviation < 0 and soc < soc_hi:
                extra_chg = min(p_left, -deviation, max(0, (soc_hi - soc) / eta_s))
                soc += eta_s * extra_chg

        total_chg = chg + extra_chg
        total_dis = dis + extra_dis
        grid = max(0, r["buy_forecast"] + total_chg - total_dis)
        cost, imb, short, sur = settle(grid, r["buy_price"], r["retailer_load"], r["price_id"], total_chg, total_dis)
        tc += cost
        ti += imb
        tsh += short
        tsur += sur
        tst += total_chg
    return tc, ti, tsur, tsh, tst


def find_id_column(da_col, price_fc):
    """The ID forecast that belongs to a DA forecast column (fc_arx_da -> fc_arx_id)."""
    if da_col is None:
        return None
    candidate = (da_col[:-3] + "_id") if da_col.endswith("_da") else da_col.replace("_da_", "_id_")
    if candidate != da_col and candidate in price_fc.columns:
        return candidate
    if da_col.startswith("fc_") and not da_col.startswith("fc_id_"):
        if "fc_id_" + da_col[3:] in price_fc.columns:
            return "fc_id_" + da_col[3:]
    suffix = da_col.replace("fc_", "").replace("_da_", "_")
    return next((c for c in price_fc.columns if c.startswith("fc_id_") and suffix in c), None)


def price_base(load_fc, price_fc, df_train, price_col, share):
    """Prices, routing and the forecast spread, shared by all load models."""
    hourly = df_train.groupby("hour").agg(avg_DA=("dayahead_price", "mean"), avg_ID=("intraday_price", "mean"))
    hourly_market = dict(zip(hourly.index, np.where(hourly["avg_DA"] <= hourly["avg_ID"], "DA", "ID")))

    id_col = find_id_column(price_col, price_fc)
    if price_col and id_col and id_col in price_fc.columns:
        spread_fc = price_fc[price_col].values.astype(float) - price_fc[id_col].values.astype(float)
    else:
        log.warning("no ID column for %s, spread strategy gets spread_fc = 0", price_col)
        spread_fc = np.zeros(len(price_fc))
        id_col = None

    base = load_fc[["datetime", "date", "hour", "load_actual"]].copy()
    base["retailer_load"] = share * base["load_actual"]
    base["price_da"] = price_fc["dayahead_price"].values
    base["price_id"] = price_fc["intraday_price"].values
    base["spread_fc"] = spread_fc
    base["market_stat"] = base["hour"].map(hourly_market)
    base["buy_price_stat"] = np.where(base["market_stat"] == "DA", base["price_da"], base["price_id"])

    if "market_ml" in price_fc.columns:
        market_ml = price_fc["market_ml"].values
    elif price_col and id_col:
        market_ml = np.where(price_fc[id_col].values < price_fc[price_col].values, "ID", "DA")
    else:
        log.warning("market_ml falls back to the actual prices (perfect routing)")
        market_ml = np.where(price_fc["intraday_price"].values < price_fc["dayahead_price"].values, "ID", "DA")
    base["market_ml"] = market_ml
    base["buy_price_ml"] = np.where(base["market_ml"] == "DA", base["price_da"], base["price_id"])
    return base


def fixed_costs(params):
    """Annual CAPEX (annuity) and O&M for one storage size."""
    r, n = params["discount_rate"], params["lifetime_years"]
    annuity = r / (1 - (1 + r) ** -n)
    capex = params["E_max_MWh"] * 1000 * params["capex_eur_per_kwh"] * annuity
    om = params["E_max_MWh"] * 1000 * params["capex_eur_per_kwh"] * params["om_pct_capex"]
    return capex, om


def run_scenario(load_fc, price_fc, df_train, cfg, surplus_mode, imbalance_factor):
    share = cfg["share"]
    load_cols = [c for c in cfg["bess_load_models"] if c in load_fc.columns]
    if not load_cols:
        raise ValueError("none of the BESS load models is in load_fc")
    price_col = cfg["bess_price_model"]
    common = bess_common(cfg)
    settle = Settlement(surplus_mode, imbalance_factor)

    # Spread threshold: 75th percentile of |DA - ID| in the training window
    threshold = float(np.percentile((df_train["dayahead_price"] - df_train["intraday_price"]).abs().dropna(), 75))
    log.info("spread threshold (p75 |DA-ID|, train): %.2f EUR/MWh", threshold)

    base = price_base(load_fc, price_fc, df_train, price_col, share)
    total_mwh = float(base["retailer_load"].sum())
    strategies = {
        "BESS_arbitrage": lambda d, p: run_arbitrage(d, p, settle),
        "BESS_balancing": lambda d, p: run_balancing(d, p, settle),
        "BESS_spread": lambda d, p: run_spread(d, p, settle, threshold),
    }

    rows = []
    for load_col in load_cols:
        data_base = base.copy()
        data_base["buy_forecast"] = share * load_fc[load_col].fillna(load_fc["fc_profile"])
        for storage in cfg["bess_configs"]:
            params = {**common, **storage}
            params["E_max_MWh"] = params["P_max_MW"] * params["duration_h"]
            capex, om = fixed_costs(params)
            grid_fee = float(params.get("grid_fee_eur_mwh", 0.0))

            for variant, price_column in [("stat", "buy_price_stat"), ("ml", "buy_price_ml")]:
                data = data_base.copy()
                data["buy_price"] = data[price_column]
                for name, strategy in strategies.items():
                    tc, ti, tsur, tsh, tst = strategy(data, params)
                    fee = tst * grid_fee
                    total = tc + fee
                    total_capex = total + (capex + om)
                    rows.append({
                        "load_model": load_col.replace("fc_", ""),
                        "method": f"{name}_{variant}",
                        "storage": storage["name"],
                        "total_cost_eur": round(total, 2),
                        "cost_eur_per_mwh": round(total / total_mwh, 4),
                        "imbalance_cost_eur": round(ti, 2),
                        "grid_fee_cost_eur": round(fee, 2),
                        "capex_annual_eur": round(capex, 2),
                        "om_annual_eur": round(om, 2),
                        "total_incl_capex_eur": round(total_capex, 2),
                        "cost_incl_capex_eur_per_mwh": round(total_capex / total_mwh, 4),
                        "surplus_mwh": round(tsur, 2),
                        "shortage_mwh": round(tsh, 2),
                        "stored_mwh": round(tst, 2),
                    })
        log.info("BESS done for %s", load_col)

    return pd.DataFrame(rows).sort_values("total_cost_eur").reset_index(drop=True)


def run(cfg, load_fc, price_fc, df_train):
    out_dir = Path(cfg["out_dir"])
    results = {}
    for sc in cfg["surplus_scenarios"]:
        res = run_scenario(load_fc, price_fc, df_train, cfg, sc["surplus_mode"], sc["imbalance_price_factor"])
        res["scenario"] = sc["label"]
        results[sc["label"]] = res
        top = res.head(20)[["load_model", "method", "storage", "cost_eur_per_mwh", "cost_incl_capex_eur_per_mwh"]]
        log.info("[%s] top 20:\n%s", sc["label"], top.to_string(index=False))
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            res.to_csv(out_dir / f"bess_results_{sc['label']}.csv", index=False)
        except Exception as e:
            log.warning("saving %s failed: %s", sc["label"], e)

    combined = pd.concat(results.values(), ignore_index=True)
    try:
        combined.to_csv(out_dir / "bess_results_all.csv", index=False)
    except Exception as e:
        log.warning("saving combined results failed: %s", e)
    return {"bess_results": results, "bess_results_all": combined}
