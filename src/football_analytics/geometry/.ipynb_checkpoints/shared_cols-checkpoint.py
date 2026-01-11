from __future__ import annotations
import numpy as np
import pandas as pd

from football_analytics.geometry.pitch import third_from_x, channel_from_y

def add_shared_geometry_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    has_loc = df["loc_x"].notna() & df["loc_y"].notna()
    df.loc[has_loc, "origin_third"] = third_from_x(df.loc[has_loc, "loc_x"])
    df.loc[has_loc, "origin_channel"] = channel_from_y(df.loc[has_loc, "loc_y"])

    is_pass = (
        (df["type"] == "Pass")
        & df["loc_x"].notna() & df["loc_y"].notna()
        & df["pass_end_x"].notna() & df["pass_end_y"].notna()
    )
    df.loc[is_pass, "pass_dx"] = df.loc[is_pass, "pass_end_x"] - df.loc[is_pass, "loc_x"]
    df.loc[is_pass, "pass_dy"] = df.loc[is_pass, "pass_end_y"] - df.loc[is_pass, "loc_y"]
    df.loc[is_pass, "pass_len"] = np.hypot(df.loc[is_pass, "pass_dx"], df.loc[is_pass, "pass_dy"])
    df.loc[is_pass, "pass_theta"] = np.arctan2(df.loc[is_pass, "pass_dy"], df.loc[is_pass, "pass_dx"])
    df.loc[is_pass, "pass_angle_deg"] = np.degrees(df.loc[is_pass, "pass_theta"])

    is_carry = (
        (df["type"] == "Carry")
        & df["loc_x"].notna() & df["loc_y"].notna()
        & df["carry_end_x"].notna() & df["carry_end_y"].notna()
    )
    df.loc[is_carry, "carry_dx"] = df.loc[is_carry, "carry_end_x"] - df.loc[is_carry, "loc_x"]
    df.loc[is_carry, "carry_dy"] = df.loc[is_carry, "carry_end_y"] - df.loc[is_carry, "loc_y"]
    df.loc[is_carry, "carry_len"] = np.hypot(df.loc[is_carry, "carry_dx"], df.loc[is_carry, "carry_dy"])
    df.loc[is_carry, "carry_theta"] = np.arctan2(df.loc[is_carry, "carry_dy"], df.loc[is_carry, "carry_dx"])
    df.loc[is_carry, "carry_angle_deg"] = np.degrees(df.loc[is_carry, "carry_theta"])

    return df

def compute_team_match_means(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["loc_x"].notna() & df["loc_y"].notna()]
    return (
        valid.groupby(["team", "match_id"])[["loc_x", "loc_y"]]
             .mean()
             .rename(columns={"loc_x":"team_mean_loc_x","loc_y":"team_mean_loc_y"})
             .reset_index()
    )
