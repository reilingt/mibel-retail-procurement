"""Runs the pipeline steps 1 to 8 in order.

    python src/run_all.py                   # everything
    python src/run_all.py --steps 6 7 8     # procurement, BESS, significance
    python src/run_all.py --data-dir /path/to/data --out-dir outputs

Steps 4 and 5 resume from load_fc.csv / price_fc.csv in out_dir if present. Steps 6 to 8
read those CSVs when 4 and 5 are not in the run. Step 1 always runs.
"""
import argparse
import importlib
import logging
import sys
from pathlib import Path

import pandas as pd

# Make sure the step files next to this script are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import build_cfg

# step number -> module file (without .py)
STEP_MODULES = {
    1: "01_data_prep",
    2: "02_eda_load",
    3: "03_eda_price",
    4: "04_load_forecast",
    5: "05_price_forecast",
    6: "06_procurement",
    7: "07_bess",
    8: "08_significance",
}


def load_step(n):
    return importlib.import_module(STEP_MODULES[n])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--steps", nargs="+", type=int, default=list(range(1, 9)), choices=range(1, 9))
    parser.add_argument("--data-dir")
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    overrides = {k: v for k, v in {"data_dir": args.data_dir, "out_dir": args.out_dir}.items() if v}
    cfg = build_cfg(**overrides)
    steps = set(args.steps)

    # Step 1 always runs: every later step needs the feature table it returns.
    prep = load_step(1).run(cfg)
    df, df_train, df_test = prep["df"], prep["df_train"], prep["df_test"]

    if 2 in steps:
        load_step(2).run(cfg, df, df_train, df_test)
    if 3 in steps:
        load_step(3).run(cfg, df_train, df_test)

    if 4 in steps:
        load_fc = load_step(4).run(cfg, df, df_train, df_test, prep["shap_sets_load"])["load_fc"]
    elif steps & {6, 7, 8}:
        load_fc = pd.read_csv(f"{cfg['out_dir']}/load_fc.csv", parse_dates=["datetime"])

    if 5 in steps:
        price_fc = load_step(5).run(cfg, df, df_train, df_test,
                                    prep["shap_sets_price"], prep["shap_sets_id_price"])["price_fc"]
    elif steps & {6, 7, 8}:
        price_fc = pd.read_csv(f"{cfg['out_dir']}/price_fc.csv", parse_dates=["datetime"])

    if 6 in steps:
        load_step(6).run(cfg, load_fc, price_fc, df_train)
    if 7 in steps:
        load_step(7).run(cfg, load_fc, price_fc, df_train)
    if 8 in steps:
        load_step(8).run(cfg, load_fc, price_fc, df_train)


if __name__ == "__main__":
    main()
