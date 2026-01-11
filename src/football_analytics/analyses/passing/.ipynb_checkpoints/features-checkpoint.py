from __future__ import annotations

import numpy as np
import pandas as pd

from football_analytics.config import X_ATT
from football_analytics.geometry.pitch import (
    per90,
    safe_pct,  # we will avoid using safe_pct where we want NaNs for denom=0
    third_from_x,
    channel_from_y,
    zone_fractions,
    thirds_from_zones,
    channels_from_zones,
    angle_bucket_masks,
    circular_angle_stats,
)

NA = float("nan")


def _get_minutes(player_data: pd.DataFrame) -> float:
    """Extract total tournament minutes for per90 calculations."""
    mins = (
        player_data["total_tourn_minutes"].dropna()
        if "total_tourn_minutes" in player_data.columns
        else pd.Series(dtype=float)
    )
    return float(mins.max()) if not mins.empty else 0.0


def _get_pass_df_and_minutes(player_data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df = (
        player_data[player_data["type"] == "Pass"].copy()
        if "type" in player_data.columns
        else player_data.iloc[0:0].copy()
    )
    minutes = _get_minutes(player_data)
    return df, minutes


def _availability_flags(df_pass: pd.DataFrame, minutes: float) -> dict:
    """Flags to make NaNs interpretable downstream (DQA + ML)."""
    has_pass_events = int(not df_pass.empty)

    def _has_cols(cols: list[str]) -> int:
        return int(has_pass_events and all(c in df_pass.columns for c in cols))

    return {
        "has_pass_events": has_pass_events,
        "has_minutes": int(has_pass_events and minutes > 0),
        "has_pass_lengths": _has_cols(["pass_length"]),
        "has_pass_end_locations": _has_cols(["loc_x", "loc_y", "pass_end_x", "pass_end_y"]),
        "has_pass_dx": _has_cols(["pass_dx"]),
        "has_pass_angles": _has_cols(["pass_angle_deg"]),
        "has_pass_outcomes": _has_cols(["pass_outcome"]),
        "has_pass_height_body": _has_cols(["pass_height", "pass_body_part"]),
        "has_under_pressure": _has_cols(["under_pressure"]),
    }


def _per90_or_nan(count: int, minutes: float) -> float:
    """per90 wrapper that returns NaN when minutes are invalid."""
    if minutes <= 0:
        return NA
    return float(per90(count, minutes))


def _core_passing_per90(df: pd.DataFrame, minutes: float) -> dict:
    # Counts become 0 when no passes; rates (per90) are NaN if minutes invalid.
    if df.empty:
        return {
            "passes_per_90": 0.0,
            "short_passes_per_90": 0.0,
            "medium_passes_per_90": 0.0,
            "long_passes_per_90": 0.0,
        }

    length = df["pass_length"] if "pass_length" in df.columns else pd.Series(NA, index=df.index)
    length = length.dropna()

    # If pass_length missing entirely, we can still compute total passes_per_90,
    # but the short/med/long split becomes unknown.
    total = len(df)
    out = {
        "passes_per_90": _per90_or_nan(total, minutes),
        "short_passes_per_90": NA,
        "medium_passes_per_90": NA,
        "long_passes_per_90": NA,
    }

    if length.empty:
        return out

    short_mask  = length < 15
    medium_mask = (length >= 15) & (length < 30)
    long_mask   = length >= 30

    out.update({
        "short_passes_per_90": _per90_or_nan(int(short_mask.sum()),  minutes),
        "medium_passes_per_90": _per90_or_nan(int(medium_mask.sum()), minutes),
        "long_passes_per_90": _per90_or_nan(int(long_mask.sum()),   minutes),
    })
    return out


def _specialist_passes_per90(df: pd.DataFrame, minutes: float) -> dict:
    if df.empty:
        return {
            "crosses_per90": 0.0,
            "switches_per90": 0.0,
            "throughballs_per90": 0.0,
            "cutbacks_per90": 0.0,
            "backheels_per90": 0.0,
            "passes_under_pressure_per90": 0.0,
            "key_passes_per90": 0.0,
            "assists_per90": 0.0,
            "pct_passes_under_pressure": NA,  # not applicable with no passes
        }

    # These masks will be all-False if the column is missing → but that would be "unknown",
    # not "0". So if the columns are missing, return NaN for those features.
    def _bool_col(col: str) -> pd.Series | None:
        return df[col] == True if col in df.columns else None

    crosses = _bool_col("pass_cross")
    switches = _bool_col("pass_switch")
    through = _bool_col("pass_through_ball")
    cutbacks = _bool_col("pass_cut_back")
    backheels = _bool_col("pass_backheel")
    under_pressure = _bool_col("under_pressure")

    key_passes_mask = df["pass_shot_assist"].notna() if "pass_shot_assist" in df.columns else None
    assists_mask = df["pass_goal_assist"].notna() if "pass_goal_assist" in df.columns else None

    def _count_per90(mask: pd.Series | None) -> float:
        if mask is None:
            return NA
        return _per90_or_nan(int(mask.sum()), minutes)

    out = {
        "crosses_per90": _count_per90(crosses),
        "switches_per90": _count_per90(switches),
        "throughballs_per90": _count_per90(through),
        "cutbacks_per90": _count_per90(cutbacks),
        "backheels_per90": _count_per90(backheels),
        "passes_under_pressure_per90": _count_per90(under_pressure),
        "key_passes_per90": _count_per90(key_passes_mask),
        "assists_per90": _count_per90(assists_mask),
        "pct_passes_under_pressure": NA,
    }

    if under_pressure is not None:
        out["pct_passes_under_pressure"] = float(under_pressure.mean()) if len(df) > 0 else NA

    return out


def _passing_length_profile(df: pd.DataFrame) -> dict:
    # Undefined if no passes or no lengths
    if df.empty or "pass_length" not in df.columns:
        return {
            "avg_pass_length": NA,
            "std_pass_length": NA,
            "pct_short_pass": NA,
            "pct_med_pass": NA,
            "pct_long_pass": NA,
        }

    length = df["pass_length"].dropna()
    if length.empty:
        return {
            "avg_pass_length": NA,
            "std_pass_length": NA,
            "pct_short_pass": NA,
            "pct_med_pass": NA,
            "pct_long_pass": NA,
        }

    short_mask  = length < 15
    medium_mask = (length >= 15) & (length < 30)
    long_mask   = length >= 30

    return {
        "avg_pass_length": float(length.mean()),
        "std_pass_length": float(length.std(ddof=0)),
        "pct_short_pass": float(short_mask.mean()),
        "pct_med_pass": float(medium_mask.mean()),
        "pct_long_pass": float(long_mask.mean()),
    }


def _passes_height_body_profile(df: pd.DataFrame) -> dict:
    base = {
        "pct_ground_pass": NA,
        "pct_low_pass": NA,
        "pct_high_pass": NA,
        "pct_left_foot_pass": NA,
        "pct_right_foot_pass": NA,
        "pct_head_pass": NA,
        "pct_foot_pass": NA,
        "pct_other_pass": NA,
        "pct_keeper_arm_pass": NA,
    }
    if df.empty or "pass_height" not in df.columns or "pass_body_part" not in df.columns:
        return base

    d = df[df["pass_height"].notna() & df["pass_body_part"].notna()]
    if d.empty:
        return base

    height = d["pass_height"]
    body = d["pass_body_part"].fillna("Other")
    total = len(d)

    # Using safe_pct here would return 0 when denom=0, but denom isn't 0 here.
    ground = float((height == "Ground Pass").mean()) if total > 0 else NA
    low = float((height == "Low Pass").mean()) if total > 0 else NA
    high = float((height == "High Pass").mean()) if total > 0 else NA

    left = float((body == "Left Foot").mean()) if total > 0 else NA
    right = float((body == "Right Foot").mean()) if total > 0 else NA
    head = float((body == "Head").mean()) if total > 0 else NA
    other = float((body == "Other").mean()) if total > 0 else NA
    keeper = float((body == "Keeper Arm").mean()) if total > 0 else NA
    foot = (left + right) if (not np.isnan(left) and not np.isnan(right)) else NA

    return {
        "pct_ground_pass": ground,
        "pct_low_pass": low,
        "pct_high_pass": high,
        "pct_left_foot_pass": left,
        "pct_right_foot_pass": right,
        "pct_head_pass": head,
        "pct_foot_pass": foot,
        "pct_other_pass": other,
        "pct_keeper_arm_pass": keeper,
    }


def _directional_passing_profile(df: pd.DataFrame) -> dict:
    if df.empty or "pass_dx" not in df.columns:
        return {
            "pct_progressive_passes": NA,
            "pct_lateral_passes": NA,
            "pct_defensive_passes": NA,
        }

    threshold = 10.0
    dx = df["pass_dx"].dropna()
    if dx.empty:
        return {
            "pct_progressive_passes": NA,
            "pct_lateral_passes": NA,
            "pct_defensive_passes": NA,
        }

    progressive = dx > threshold
    defensive = dx < -threshold
    lateral = ~progressive & ~defensive

    return {
        "pct_progressive_passes": float(progressive.mean()),
        "pct_lateral_passes": float(lateral.mean()),
        "pct_defensive_passes": float(defensive.mean()),
    }


def _percentage_passes_profile(df: pd.DataFrame) -> dict:
    # Return NaNs if we can't compute (unknown), not zeros.
    base = {
        "pct_pass_from_def_third": NA,
        "pct_pass_from_mid_third": NA,
        "pct_pass_from_att_third": NA,
        "pct_pass_from_left_channel": NA,
        "pct_pass_from_central_channel": NA,
        "pct_pass_from_right_channel": NA,
        "pct_pass_to_def_third": NA,
        "pct_pass_to_mid_third": NA,
        "pct_pass_to_left_channel": NA,
        "pct_pass_to_central_channel": NA,
        "pct_pass_to_right_channel": NA,
        "pct_passes_final_third": NA,
        "pct_passes_into_box": NA,
    }
    for z in ["dl", "dc", "dr", "ml", "mc", "mr", "al", "ac", "ar"]:
        base[f"pct_pass_from_zone_{z}"] = NA
        base[f"pct_pass_to_zone_{z}"] = NA

    needed = ["loc_x", "loc_y", "pass_end_x", "pass_end_y"]
    if df.empty or any(c not in df.columns for c in needed):
        return base

    d = df[df["loc_x"].notna() & df["loc_y"].notna() & df["pass_end_x"].notna() & df["pass_end_y"].notna()]
    if d.empty:
        return base

    x0, y0 = d["loc_x"], d["loc_y"]
    x1, y1 = d["pass_end_x"], d["pass_end_y"]

    zones_from = zone_fractions(x0, y0)
    thirds_from = thirds_from_zones(zones_from)
    channels_from = channels_from_zones(zones_from)

    zones_to = zone_fractions(x1, y1)
    thirds_to = thirds_from_zones(zones_to)
    channels_to = channels_from_zones(zones_to)

    pct_final_third = float((x1 >= X_ATT).mean())
    pct_into_box = float(((x1 >= 102) & (y1 >= 18) & (y1 <= 62)).mean())

    result = {
        "pct_pass_from_def_third": thirds_from["def"],
        "pct_pass_from_mid_third": thirds_from["mid"],
        "pct_pass_from_att_third": thirds_from["att"],
        "pct_pass_from_left_channel": channels_from["left"],
        "pct_pass_from_central_channel": channels_from["centre"],
        "pct_pass_from_right_channel": channels_from["right"],
        "pct_pass_to_def_third": thirds_to["def"],
        "pct_pass_to_mid_third": thirds_to["mid"],
        "pct_pass_to_left_channel": channels_to["left"],
        "pct_pass_to_central_channel": channels_to["centre"],
        "pct_pass_to_right_channel": channels_to["right"],
        "pct_passes_final_third": pct_final_third,
        "pct_passes_into_box": pct_into_box,
    }

    for z in zones_from.keys():
        result[f"pct_pass_from_zone_{z}"] = zones_from[z]
        result[f"pct_pass_to_zone_{z}"] = zones_to[z]

    return result


def _pass_transition_profile(df: pd.DataFrame) -> dict:
    base = {
        "pct_pass_def_to_mid": NA, "pct_pass_def_to_att": NA, "pct_pass_mid_to_att": NA,
        "pct_pass_mid_to_mid": NA, "pct_pass_att_to_mid": NA, "pct_pass_def_to_def": NA, "pct_pass_att_to_att": NA,
        "pct_pass_left_to_centre": NA, "pct_pass_left_to_right": NA, "pct_pass_right_to_centre": NA,
        "pct_pass_right_to_left": NA, "pct_pass_centre_to_left": NA, "pct_pass_centre_to_right": NA,
        "pct_pass_wide_to_box": NA, "pct_pass_centre_to_box": NA, "pct_pass_def_to_box": NA,
    }

    needed = ["loc_x", "loc_y", "pass_end_x", "pass_end_y"]
    if df.empty or any(c not in df.columns for c in needed):
        return base

    d = df[df["loc_x"].notna() & df["loc_y"].notna() & df["pass_end_x"].notna() & df["pass_end_y"].notna()].copy()
    if d.empty:
        return base

    d["origin_third"] = third_from_x(d["loc_x"])
    d["dest_third"] = third_from_x(d["pass_end_x"])
    d["origin_channel"] = channel_from_y(d["loc_y"])
    d["dest_channel"] = channel_from_y(d["pass_end_y"])

    total = len(d)
    third_pairs = d.groupby(["origin_third", "dest_third"]).size() / total
    channel_pairs = d.groupby(["origin_channel", "dest_channel"]).size() / total

    def get_third(o, t) -> float:
        return float(third_pairs.get((o, t), 0.0))

    def get_chan(o, t) -> float:
        return float(channel_pairs.get((o, t), 0.0))

    x_end, y_end = d["pass_end_x"], d["pass_end_y"]
    in_box = (x_end >= 102) & (y_end >= 18) & (y_end <= 62)

    wide_origin = d["origin_channel"].isin(["left", "right"])
    centre_origin = d["origin_channel"] == "centre"
    def_origin = d["origin_third"] == "def"

    return {
        "pct_pass_def_to_mid": get_third("def", "mid"),
        "pct_pass_def_to_att": get_third("def", "att"),
        "pct_pass_mid_to_att": get_third("mid", "att"),
        "pct_pass_mid_to_mid": get_third("mid", "mid"),
        "pct_pass_att_to_mid": get_third("att", "mid"),
        "pct_pass_def_to_def": get_third("def", "def"),
        "pct_pass_att_to_att": get_third("att", "att"),

        "pct_pass_left_to_centre": get_chan("left", "centre"),
        "pct_pass_left_to_right": get_chan("left", "right"),
        "pct_pass_right_to_centre": get_chan("right", "centre"),
        "pct_pass_right_to_left": get_chan("right", "left"),
        "pct_pass_centre_to_left": get_chan("centre", "left"),
        "pct_pass_centre_to_right": get_chan("centre", "right"),

        "pct_pass_wide_to_box": float((wide_origin & in_box).mean()),
        "pct_pass_centre_to_box": float((centre_origin & in_box).mean()),
        "pct_pass_def_to_box": float((def_origin & in_box).mean()),
    }


def _angle_volume_features(angle_deg: pd.Series, minutes: float, prefix: str = "passes") -> dict:
    masks = angle_bucket_masks(angle_deg)
    total = int(len(angle_deg))
    result = {}

    for code, mask in masks.items():
        bucket_count = int(mask.sum())
        result[f"ttl_{prefix}_{code}"] = float(bucket_count)

        # pct undefined if no angles
        result[f"pct_{prefix}_{code}"] = (float(bucket_count / total) if total > 0 else NA)

        # per90 undefined if minutes invalid; if no angles at all, treat as NA not 0 (unknown)
        if total == 0:
            result[f"{prefix}_{code}_per90"] = NA
        else:
            result[f"{prefix}_{code}_per90"] = _per90_or_nan(bucket_count, minutes)

    return result


def _angle_completion_features(angle_deg: pd.Series, completed_mask: pd.Series) -> dict:
    masks = angle_bucket_masks(angle_deg)
    result = {}
    for code, bucket_mask in masks.items():
        denom = int(bucket_mask.sum())
        # Undefined if denom==0 (no attempts in this bucket)
        if denom == 0:
            result[f"completion_pct_{code}"] = NA
        else:
            result[f"completion_pct_{code}"] = float((completed_mask & bucket_mask).sum()) / denom
    return result


def _angle_height_features(df: pd.DataFrame, angle_deg: pd.Series) -> dict:
    if "pass_height" not in df.columns:
        return {}

    masks = angle_bucket_masks(angle_deg)
    height = df["pass_height"]
    result = {}

    for code, bucket_mask in masks.items():
        denom = int(bucket_mask.sum())
        if denom == 0:
            result[f"pct_ground_{code}"] = NA
            result[f"pct_low_{code}"] = NA
            result[f"pct_high_{code}"] = NA
            continue
        h = height[bucket_mask]
        result[f"pct_ground_{code}"] = float((h == "Ground Pass").mean())
        result[f"pct_low_{code}"] = float((h == "Low Pass").mean())
        result[f"pct_high_{code}"] = float((h == "High Pass").mean())
    return result


def _angle_length_features(df: pd.DataFrame, angle_deg: pd.Series) -> dict:
    if "pass_length" not in df.columns:
        return {}
    masks = angle_bucket_masks(angle_deg)
    length = df["pass_length"]
    result = {}
    for code, bucket_mask in masks.items():
        # Undefined if no events in bucket
        result[f"avg_length_{code}"] = float(length[bucket_mask].mean()) if bucket_mask.any() else NA
    return result


def _angle_body_features(df: pd.DataFrame, angle_deg: pd.Series) -> dict:
    if "pass_body_part" not in df.columns:
        return {}

    masks = angle_bucket_masks(angle_deg)
    body = df["pass_body_part"].fillna("Other")
    result = {}

    for code, bucket_mask in masks.items():
        denom = int(bucket_mask.sum())
        if denom == 0:
            result[f"pct_left_foot_{code}"] = NA
            result[f"pct_right_foot_{code}"] = NA
            result[f"pct_head_{code}"] = NA
            result[f"pct_foot_{code}"] = NA
            result[f"pct_other_{code}"] = NA
            result[f"pct_keeper_arm_{code}"] = NA
            continue

        b = body[bucket_mask]
        left = float((b == "Left Foot").mean())
        right = float((b == "Right Foot").mean())
        head = float((b == "Head").mean())
        other = float((b == "Other").mean())
        keeper = float((b == "Keeper Arm").mean())
        foot = left + right

        result[f"pct_left_foot_{code}"] = left
        result[f"pct_right_foot_{code}"] = right
        result[f"pct_head_{code}"] = head
        result[f"pct_foot_{code}"] = foot
        result[f"pct_other_{code}"] = other
        result[f"pct_keeper_arm_{code}"] = keeper

    return result


def _angle_based_passing_profile(df: pd.DataFrame, minutes: float) -> dict:
    # We only compute angle-family features when we have angle data.
    if df.empty or "pass_angle_deg" not in df.columns:
        # No passes or no angle column -> "unknown angle family"
        out = _angle_volume_features(pd.Series(dtype=float), minutes, prefix="passes")
        out.update({
            "pass_angle_mean_overall": NA, "pass_angle_var_overall": NA,
            "pass_angle_mean_def_third": NA, "pass_angle_var_def_third": NA,
            "pass_angle_mean_mid_third": NA, "pass_angle_var_mid_third": NA,
            "pass_angle_mean_att_third": NA, "pass_angle_var_att_third": NA,
            "pass_angle_mean_left_channel": NA, "pass_angle_var_left_channel": NA,
            "pass_angle_mean_centre_channel": NA, "pass_angle_var_centre_channel": NA,
            "pass_angle_mean_right_channel": NA, "pass_angle_var_right_channel": NA,
        })
        return out

    d = df[df["pass_angle_deg"].notna()].copy()
    if d.empty:
        # Column exists, but no actual angle values -> still unknown
        out = _angle_volume_features(pd.Series(dtype=float), minutes, prefix="passes")
        out.update({
            "pass_angle_mean_overall": NA, "pass_angle_var_overall": NA,
            "pass_angle_mean_def_third": NA, "pass_angle_var_def_third": NA,
            "pass_angle_mean_mid_third": NA, "pass_angle_var_mid_third": NA,
            "pass_angle_mean_att_third": NA, "pass_angle_var_att_third": NA,
            "pass_angle_mean_left_channel": NA, "pass_angle_var_left_channel": NA,
            "pass_angle_mean_centre_channel": NA, "pass_angle_var_centre_channel": NA,
            "pass_angle_mean_right_channel": NA, "pass_angle_var_right_channel": NA,
        })
        return out

    angle_deg = d["pass_angle_deg"]
    theta = d["pass_theta"] if "pass_theta" in d.columns else pd.Series(np.radians(angle_deg), index=d.index)

    # Completion mask: if pass_outcome missing, completion by angle is unknown (NaN)
    if "pass_outcome" in d.columns:
        completed_mask = d["pass_outcome"].isna()
    else:
        completed_mask = pd.Series([False] * len(d), index=d.index)

    result = {}
    result.update(_angle_volume_features(angle_deg, minutes, prefix="passes"))
    result.update(_angle_length_features(d, angle_deg))
    # Only meaningful if outcomes exist; otherwise these will still compute but be misleading.
    if "pass_outcome" in d.columns:
        result.update(_angle_completion_features(angle_deg, completed_mask))
    else:
        # Fill completion_pct_* with NaN consistently
        for code in angle_bucket_masks(angle_deg).keys():
            result[f"completion_pct_{code}"] = NA

    result.update(_angle_height_features(d, angle_deg))
    result.update(_angle_body_features(d, angle_deg))

    origin_third = d["origin_third"] if "origin_third" in d.columns else third_from_x(d["loc_x"])
    origin_channel = d["origin_channel"] if "origin_channel" in d.columns else channel_from_y(d["loc_y"])

    def_mask = origin_third == "def"
    mid_mask = origin_third == "mid"
    att_mask = origin_third == "att"
    left_mask = origin_channel == "left"
    cen_mask = origin_channel == "centre"
    right_mask = origin_channel == "right"

    mean_o, var_o = circular_angle_stats(theta)
    mean_d, var_d = circular_angle_stats(theta, def_mask)
    mean_m, var_m = circular_angle_stats(theta, mid_mask)
    mean_a, var_a = circular_angle_stats(theta, att_mask)

    mean_l, var_l = circular_angle_stats(theta, left_mask)
    mean_c, var_c = circular_angle_stats(theta, cen_mask)
    mean_r, var_r = circular_angle_stats(theta, right_mask)

    result.update({
        "pass_angle_mean_overall": mean_o, "pass_angle_var_overall": var_o,
        "pass_angle_mean_def_third": mean_d, "pass_angle_var_def_third": var_d,
        "pass_angle_mean_mid_third": mean_m, "pass_angle_var_mid_third": var_m,
        "pass_angle_mean_att_third": mean_a, "pass_angle_var_att_third": var_a,
        "pass_angle_mean_left_channel": mean_l, "pass_angle_var_left_channel": var_l,
        "pass_angle_mean_centre_channel": mean_c, "pass_angle_var_centre_channel": var_c,
        "pass_angle_mean_right_channel": mean_r, "pass_angle_var_right_channel": var_r,
    })

    return result


def calculate_passing_features(player_data: pd.DataFrame) -> dict:
    df_pass, minutes = _get_pass_df_and_minutes(player_data)

    result = {}
    # Put flags first so they’re always present
    result.update(_availability_flags(df_pass, minutes))

    result.update(_core_passing_per90(df_pass, minutes))
    result.update(_specialist_passes_per90(df_pass, minutes))
    result.update(_passing_length_profile(df_pass))
    result.update(_passes_height_body_profile(df_pass))
    result.update(_directional_passing_profile(df_pass))
    result.update(_percentage_passes_profile(df_pass))
    result.update(_pass_transition_profile(df_pass))
    result.update(_angle_based_passing_profile(df_pass, minutes))
    return result

def passing_feature_columns() -> list[str]:
    dummy = pd.DataFrame({"type": pd.Series(dtype="object")})
    return list(calculate_passing_features(dummy).keys())
