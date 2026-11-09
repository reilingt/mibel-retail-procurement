# Build the public dataset: keep every real column except the two whose source
# licence forbids redistribution (TTF gas via Yahoo/ICE, EUA carbon via ICAP/ICE).
# Those two are replaced by synthetic series with matched level, drift and
# volatility, so the pipeline behaves the same but no real market values ship.
# See DATA_LICENSES.md. Deterministic via --seed.
import argparse
import numpy as np
import pandas as pd

SYNTHETIC_COLS = ["ttf_eur_mwh", "carbon_eur_t"]


def synth_commodity(real, dates, rng):
    # real is the hourly column; collapse to one value per day and model the daily
    # log-price as a mean-reverting AR(1) matched to the real series' log mean,
    # log std and daily autocorrelation, then broadcast back to hourly. This keeps
    # the level distribution and persistence realistic while producing a fresh path.
    day = pd.Series(real.values, index=dates).groupby(dates.normalize()).first()
    logp = np.log(day.clip(lower=0.01))
    mu, sigma = logp.mean(), logp.std()
    phi = float(np.clip(logp.autocorr(1), 0.0, 0.999))
    eps_sd = sigma * np.sqrt(max(1e-6, 1.0 - phi**2))
    lo, hi = day.min() * 0.7, day.max() * 1.3

    l = np.empty(len(day))
    l[0] = mu
    noise = rng.normal(0.0, eps_sd, len(day))
    for t in range(1, len(day)):
        l[t] = mu + phi * (l[t - 1] - mu) + noise[t]
    synth = np.exp(l)
    # lift onto the real arithmetic level (log-mean understates skewed prices)
    synth *= day.mean() / synth.mean()
    synth_day = pd.Series(np.clip(synth, lo, hi), index=day.index)

    mapped = pd.Series(dates.normalize()).map(synth_day).values
    return np.round(mapped, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="real merged CSV")
    ap.add_argument("--out", required=True, help="public CSV to write")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.inp, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    dates = pd.DatetimeIndex(df["datetime"])

    for col in SYNTHETIC_COLS:
        if col in df.columns:
            df[col] = synth_commodity(df[col], dates, rng)
            print(f"synthesised {col}")
        else:
            print(f"warning: {col} not in input, skipped")

    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({df.shape[0]} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
