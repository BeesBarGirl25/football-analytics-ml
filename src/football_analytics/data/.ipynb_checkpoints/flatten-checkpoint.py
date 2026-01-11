from __future__ import annotations
import numpy as np
import pandas as pd

def _split_location_col(df: pd.DataFrame, col: str, x_name: str, y_name: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    arr = df[col].apply(lambda v: v if isinstance(v, (list, tuple)) else [np.nan, np.nan])
    df[x_name] = arr.apply(lambda v: v[0] if len(v) > 0 else np.nan)
    df[y_name] = arr.apply(lambda v: v[1] if len(v) > 1 else np.nan)
    return df.drop(columns=[col])

def flatten_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    loc_map = {
        "location": ("loc_x", "loc_y"),
        "pass_end_location": ("pass_end_x", "pass_end_y"),
        "carry_end_location": ("carry_end_x", "carry_end_y"),
        "shot_end_location": ("shot_end_x", "shot_end_y"),
        "goalkeeper_end_location": ("gk_end_x", "gk_end_y"),
    }

    for col, (x_name, y_name) in loc_map.items():
        df = _split_location_col(df, col, x_name, y_name)

    if "50_50" in df.columns:
        df["50_50_outcome"] = df["50_50"].apply(lambda d: d.get("outcome", {}).get("name") if isinstance(d, dict) else None)
        df["50_50_counterpress"] = df["50_50"].apply(lambda d: d.get("counterpress") if isinstance(d, dict) else None)
        df = df.drop(columns=["50_50"])

    return df
