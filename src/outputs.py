"""Write result tables to out_dir as CSVs."""
from pathlib import Path

import pandas as pd


def save(tables, out_dir, prefix):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            keep_index = not isinstance(table.index, pd.RangeIndex)
            table.to_csv(out / f"{prefix}_{name}.csv", index=keep_index)
