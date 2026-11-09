import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def step(name):
    # step files start with a digit, so they are imported by name
    return importlib.import_module(name)


@pytest.fixture
def hourly_frame():
    # three weeks of hourly data with the renamed pipeline columns
    rng = np.random.default_rng(0)
    n = 24 * 21
    df = pd.DataFrame({"datetime": pd.date_range("2023-03-01", periods=n, freq="h")})
    for col in ["load", "dayahead_price", "intraday_price", "da_price_es", "id_price_es",
                "wind", "solar", "flow_net_MW", "ttf", "eua", "gen_gas", "gen_coal",
                "hydro_ror", "congestion", "wind_fc", "solar_fc"]:
        df[col] = rng.normal(100, 10, n)
    return df
