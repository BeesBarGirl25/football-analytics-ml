from __future__ import annotations

import numpy as np
import pandas as pd

from football_analytics.config import (
    Y_LEFT_WING, Y_LEFT_HALF, Y_RIGHT_HALF, Y_RIGHT_WING,
    X_DEEP_DEF, X_BUILD_UP, X_MIDFIELD, X_ADV_MID, X_FINAL_THIRD,
)
from football_analytics.geometry.pitch import (
    zone_fractions,
    thirds_from_zones,
    channels_from_zones,
)
from football_analytics.geometry.shared_cols import compute_team_match_means
from football_analytics.analyses.passing.features import calculate_passing_features
from football_analytics.analyses.carrying.features import calculate_carry_actions_features


def calculate_percentage_occupancies(player_data: pd.DataFrame) -> dict:
    df = player_data[player_data["loc_x"].notna() & player_data["loc_y"].notna()]
    if df.empty:
        return {
            "def_third_pct": 0.0, "mid_third_pct": 0.0, "att_third_pct": 0.0,
            "left_channel_pct": 0.0, "mid_channel_pct": 0.0, "right_channel_pct": 0.0,
        }

    zones = zone_fractions(df["loc_x"], df["loc_y"])
    thirds = thirds_from_zones(zones)
    channels = channels_from_zones(zones)

    return {
        "def_third_pct": thirds["def"],
        "mid_third_pct": thirds["mid"],
        "att_third_pct": thirds["att"],
        "left_channel_pct": channels["left"],
        "mid_channel_pct": channels["centre"],
        "right_channel_pct": channels["right"],
    }


def calculate_percentage_zones(player_data: pd.DataFrame) -> dict:
    df = player_data[player_data["loc_x"].notna() & player_data["loc_y"].notna()]
    if df.empty:
        return {f"pct_zone_{z}": 0.0 for z in ["dl","dc","dr","ml","mc","mr","al","ac","ar"]}

    zones = zone_fractions(df["loc_x"], df["loc_y"])
    return {f"pct_zone_{k}": zones[k] for k in zones.keys()}


def calculate_shot_features(player_data: pd.DataFrame) -> dict:
    df = player_data[
        (player_data["type"] == "Shot")
        & player_data["loc_x"].notna()
        & player_data["loc_y"].notna()
    ]
    if df.empty:
        return {"mean_shot_loc_x": np.nan, "mean_shot_loc_y": np.nan, "pct_shots_in_box": 0.0}

    mean_x = float(df["loc_x"].mean())
    mean_y = float(df["loc_y"].mean())
    in_box = (df["loc_x"] >= 102) & (df["loc_y"].between(18, 62))
    return {
        "mean_shot_loc_x": mean_x,
        "mean_shot_loc_y": mean_y,
        "pct_shots_in_box": float(in_box.mean()),
    }


def calculate_statistical_features(player_data: pd.DataFrame) -> dict:
    df = player_data[player_data["loc_x"].notna() & player_data["loc_y"].notna()]
    if df.empty:
        return {"var_x_loc": 0.0, "var_y_loc": 0.0, "entropy": 0.0}

    var_x = float(np.var(df["loc_x"]))
    var_y = float(np.var(df["loc_y"]))

    zones = zone_fractions(df["loc_x"], df["loc_y"])
    p = np.array(list(zones.values()))
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum()) if len(p) else 0.0

    return {"var_x_loc": var_x, "var_y_loc": var_y, "entropy": entropy}


def calculate_relative_mean_position(player_data: pd.DataFrame, team_match_means: pd.DataFrame) -> dict:
    df = player_data[player_data["loc_x"].notna() & player_data["loc_y"].notna()]
    if df.empty:
        return {"relative_mean_loc_x": 0.0, "relative_mean_loc_y": 0.0}

    merged = df.merge(team_match_means, on=["team", "match_id"], how="left")
    rel_x = (merged["loc_x"] - merged["team_mean_loc_x"]).mean()
    rel_y = (merged["loc_y"] - merged["team_mean_loc_y"]).mean()
    return {"relative_mean_loc_x": float(rel_x), "relative_mean_loc_y": float(rel_y)}


def calculate_halfspace_features(player_data: pd.DataFrame) -> dict:
    df = player_data[player_data["loc_y"].notna()]
    if df.empty:
        return {
            "pct_actions_left_halfspace": 0.0,
            "pct_actions_central_lane": 0.0,
            "pct_actions_right_halfspace": 0.0,
        }

    y = df["loc_y"]
    left_half  = (y >= Y_LEFT_WING) & (y < Y_LEFT_HALF)
    central    = (y >= Y_LEFT_HALF) & (y < Y_RIGHT_HALF)
    right_half = (y >= Y_RIGHT_HALF) & (y < Y_RIGHT_WING)

    return {
        "pct_actions_left_halfspace": float(left_half.mean()),
        "pct_actions_central_lane": float(central.mean()),
        "pct_actions_right_halfspace": float(right_half.mean()),
    }


def calculate_depth_lane_features(player_data: pd.DataFrame) -> dict:
    df = player_data[player_data["loc_x"].notna()]
    if df.empty:
        return {
            "pct_actions_deep_def": 0.0,
            "pct_actions_build_up": 0.0,
            "pct_actions_midfield_band": 0.0,
            "pct_actions_adv_mid": 0.0,
            "pct_actions_final_third_lane": 0.0,
        }

    x = df["loc_x"]
    deep_def  = (x < X_DEEP_DEF)
    build_up  = (x >= X_DEEP_DEF) & (x < X_BUILD_UP)
    midfield  = (x >= X_BUILD_UP) & (x < X_MIDFIELD)
    adv_mid   = (x >= X_MIDFIELD) & (x < X_ADV_MID)
    final_3rd = (x >= X_ADV_MID) & (x <= X_FINAL_THIRD)

    return {
        "pct_actions_deep_def": float(deep_def.mean()),
        "pct_actions_build_up": float(build_up.mean()),
        "pct_actions_midfield_band": float(midfield.mean()),
        "pct_actions_adv_mid": float(adv_mid.mean()),
        "pct_actions_final_third_lane": float(final_3rd.mean()),
    }


def build_player_level_df(events: pd.DataFrame) -> pd.DataFrame:
    key = "player_id" if "player_id" in events.columns else "player"
    team_match_means = compute_team_match_means(events)

    rows = []
    for k, g in events.groupby(key, dropna=False):
        occupancies = calculate_percentage_occupancies(g)
        zones       = calculate_percentage_zones(g)
        shots       = calculate_shot_features(g)
        statistical = calculate_statistical_features(g)
        means       = calculate_relative_mean_position(g, team_match_means)
        halfspace   = calculate_halfspace_features(g)
        depthlane   = calculate_depth_lane_features(g)

        passing_feats = calculate_passing_features(g)
        carry_feats   = calculate_carry_actions_features(g)

        # position + name (best-effort)
        if "position" in g.columns and not g["position"].dropna().empty:
            player_position = g["position"].value_counts().idxmax()
        else:
            player_position = None

        player_name = None
        if "player" in g.columns and g["player"].notna().any():
            player_name = g["player"].dropna().iloc[0]

        loc = g[g["loc_x"].notna() & g["loc_y"].notna()]

        rows.append({
            "player_key": k,
            "player": player_name,
            "player_position": player_position,
            "mean_x_loc": float(loc["loc_x"].mean()) if not loc.empty else np.nan,
            "mean_y_loc": float(loc["loc_y"].mean()) if not loc.empty else np.nan,
            **occupancies,
            **zones,
            **shots,
            **statistical,
            **means,
            **halfspace,
            **depthlane,
            **passing_feats,
            **carry_feats,
        })

    return pd.DataFrame(rows)
