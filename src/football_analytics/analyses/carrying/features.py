from __future__ import annotations

import numpy as np
import pandas as pd

from football_analytics.config import X_DEF, X_ATT
from football_analytics.geometry.pitch import (
    per90,
    channel_from_y,
    third_from_x,
    angle_bucket_masks,
    circular_angle_stats,
)


def calculate_carry_actions_features(player_data: pd.DataFrame) -> dict:
    df = player_data[
        (player_data["type"] == "Carry")
        & player_data["loc_x"].notna()
        & player_data["loc_y"].notna()
        & player_data["carry_end_x"].notna()
        & player_data["carry_end_y"].notna()
    ].copy()

    carry_threshold = 10.0
    DEF_THIRD_MAX = float(X_DEF)
    ATT_THIRD_MIN = float(X_ATT)

    PEN_AREA_X_MIN = 102.0
    PEN_AREA_Y_MIN = 18.0
    PEN_AREA_Y_MAX = 62.0

    mins_series = player_data["total_tourn_minutes"].dropna() if "total_tourn_minutes" in player_data.columns else pd.Series(dtype=float)
    minutes = float(mins_series.max()) if not mins_series.empty else 0.0

    # --- empty return template (kept as your original behaviour)
    empty_keys = [
        "carries_total","carries_per_90","carry_distance_total","carry_distance_per_90","avg_carry_length",
        "progressive_carries_total","progressive_carries_per_90","progressive_carries_pct",
        "progressive_carry_distance_total","progressive_carry_distance_per_90","progressive_carry_distance_avg",
        "backward_carries_total","backward_carries_per_90","backward_carries_pct",
        "backward_carry_distance_total","backward_carry_distance_per_90","backward_carry_distance_avg",
        "defensive_carries_total","defensive_carries_per_90","defensive_carries_pct",
        "defensive_carry_distance_total","defensive_carry_distance_per_90","defensive_carry_distance_avg",
        "offensive_carries_total","offensive_carries_per_90","offensive_carries_pct",
        "offensive_carry_distance_total","offensive_carry_distance_per_90","offensive_carry_distance_avg",
        "final_third_carries_total","final_third_carries_per_90","final_third_carries_pct",
        "final_third_carry_distance_total","final_third_carry_distance_per_90","final_third_carry_distance_avg",
        "box_entry_carries_total","box_entry_carries_per_90","box_entry_carries_pct",
        "box_entry_carry_distance_total","box_entry_carry_distance_per_90","box_entry_carry_distance_avg",
        "def_to_mid_carries_total","def_to_mid_carries_per_90","def_to_mid_carries_pct",
        "def_to_mid_carry_distance_total","def_to_mid_carry_distance_per_90","def_to_mid_carry_distance_avg",
        "def_to_attack_carries_total","def_to_attack_carries_per_90","def_to_attack_carries_pct",
        "def_to_attack_carry_distance_total","def_to_attack_carry_distance_per_90","def_to_attack_carry_distance_avg",
        "mid_to_attack_carries_total","mid_to_attack_carries_per_90","mid_to_attack_carries_pct",
        "mid_to_attack_carry_distance_total","mid_to_attack_carry_distance_per_90","mid_to_attack_carry_distance_avg",
        "mid_to_mid_carries_total","mid_to_mid_carries_per_90","mid_to_mid_carries_pct",
        "mid_to_mid_carry_distance_total","mid_to_mid_carry_distance_per_90","mid_to_mid_carry_distance_avg",
        "att_to_mid_carries_total","att_to_mid_carries_per_90","att_to_mid_carries_pct",
        "att_to_mid_carry_distance_total","att_to_mid_carry_distance_per_90","att_to_mid_carry_distance_avg",
        "def_to_def_carries_total","def_to_def_carries_per_90","def_to_def_carries_pct",
        "def_to_def_carry_distance_total","def_to_def_carry_distance_per_90","def_to_def_carry_distance_avg",
        "att_to_att_carries_total","att_to_att_carries_per_90","att_to_att_carries_pct",
        "att_to_att_carry_distance_total","att_to_att_carry_distance_per_90","att_to_att_carry_distance_avg",
        "left_to_centre_carries_total","left_to_centre_carries_per_90","left_to_centre_carries_pct",
        "left_to_centre_carry_distance_total","left_to_centre_carry_distance_per_90","left_to_centre_carry_distance_avg",
        "left_to_right_carries_total","left_to_right_carries_per_90","left_to_right_carries_pct",
        "left_to_right_carry_distance_total","left_to_right_carry_distance_per_90","left_to_right_carry_distance_avg",
        "centre_to_left_carries_total","centre_to_left_carries_per_90","centre_to_left_carries_pct",
        "centre_to_left_carry_distance_total","centre_to_left_carry_distance_per_90","centre_to_left_carry_distance_avg",
        "centre_to_right_carries_total","centre_to_right_carries_per_90","centre_to_right_carries_pct",
        "centre_to_right_carry_distance_total","centre_to_right_carry_distance_per_90","centre_to_right_carry_distance_avg",
        "right_to_centre_carries_total","right_to_centre_carries_per_90","right_to_centre_carries_pct",
        "right_to_centre_carry_distance_total","right_to_centre_carry_distance_per_90","right_to_centre_carry_distance_avg",
        "right_to_left_carries_total","right_to_left_carries_per_90","right_to_left_carries_pct",
        "right_to_left_carry_distance_total","right_to_left_carry_distance_per_90","right_to_left_carry_distance_avg",
        "carry_angle_mean_overall","carry_angle_var_overall",
        "carry_angle_mean_def_third","carry_angle_var_def_third",
        "carry_angle_mean_mid_third","carry_angle_var_mid_third",
        "carry_angle_mean_att_third","carry_angle_var_att_third",
        "carry_angle_mean_left_channel","carry_angle_var_left_channel",
        "carry_angle_mean_centre_channel","carry_angle_var_centre_channel",
        "carry_angle_mean_right_channel","carry_angle_var_right_channel",
        "offensive_progressive_carries_total","offensive_progressive_carries_per_90","offensive_progressive_carries_pct",
        "offensive_progressive_carry_distance_total","offensive_progressive_carry_distance_per_90","offensive_progressive_carry_distance_avg",
        "lateral_carries_total","lateral_carries_per_90","lateral_carries_pct",
        "lateral_carry_distance_total","lateral_carry_distance_per_90","lateral_carry_distance_avg",
        "wide_lane_carries_total","wide_lane_carries_per_90","wide_lane_carries_pct",
        "wide_lane_carry_distance_total","wide_lane_carry_distance_per_90","wide_lane_carry_distance_avg",
        "carry_direction_entropy","max_carry_length","carry_burst_index",
    ]

    if df.empty:
        return {k: 0.0 for k in empty_keys}

    start_x = df["loc_x"]; start_y = df["loc_y"]
    end_x = df["carry_end_x"]; end_y = df["carry_end_y"]

    dx = df["carry_dx"] if "carry_dx" in df.columns else (end_x - start_x)
    dy = df["carry_dy"] if "carry_dy" in df.columns else (end_y - start_y)
    length = df["carry_len"] if "carry_len" in df.columns else np.hypot(dx, dy)
    theta = df["carry_theta"] if "carry_theta" in df.columns else np.arctan2(dy, dx)
    angle_deg = df["carry_angle_deg"] if "carry_angle_deg" in df.columns else np.degrees(theta)

    progressive_mask = dx > carry_threshold
    backward_mask   = dx < -carry_threshold

    defensive_mask = start_x < DEF_THIRD_MAX
    mid_mask       = (start_x >= DEF_THIRD_MAX) & (start_x < ATT_THIRD_MIN)
    attacking_mask = start_x >= ATT_THIRD_MIN

    offensive_end_mask   = end_x >= ATT_THIRD_MIN
    box_entry_mask = (end_x >= PEN_AREA_X_MIN) & (end_y >= PEN_AREA_Y_MIN) & (end_y <= PEN_AREA_Y_MAX)

    def_to_mid_mask = defensive_mask & (end_x > DEF_THIRD_MAX) & (end_x < ATT_THIRD_MIN)
    def_to_attack_mask = defensive_mask & (end_x >= ATT_THIRD_MIN)
    mid_to_attack_mask = mid_mask & (end_x >= ATT_THIRD_MIN)
    mid_to_mid_mask = mid_mask & (end_x >= DEF_THIRD_MAX) & (end_x < ATT_THIRD_MIN)
    att_to_mid_mask = attacking_mask & (end_x >= DEF_THIRD_MAX) & (end_x < ATT_THIRD_MIN)
    def_to_def_mask = defensive_mask & (end_x < DEF_THIRD_MAX)
    att_to_att_mask = attacking_mask & (end_x >= ATT_THIRD_MIN)

    origin_channel = channel_from_y(start_y)
    dest_channel   = channel_from_y(end_y)

    left_to_centre_mask  = (origin_channel == "left")   & (dest_channel == "centre")
    left_to_right_mask   = (origin_channel == "left")   & (dest_channel == "right")
    centre_to_left_mask  = (origin_channel == "centre") & (dest_channel == "left")
    centre_to_right_mask = (origin_channel == "centre") & (dest_channel == "right")
    right_to_centre_mask = (origin_channel == "right")  & (dest_channel == "centre")
    right_to_left_mask   = (origin_channel == "right")  & (dest_channel == "left")

    offensive_progressive_mask = attacking_mask & progressive_mask
    lateral_mask = (np.abs(angle_deg) >= 67.5) & (np.abs(angle_deg) <= 112.5)
    wide_lane_mask = (origin_channel != "centre") | (dest_channel != "centre")

    carries_total = int(len(df))
    carries_per_90 = per90(carries_total, minutes)

    carry_distance_total = float(length.sum())
    carry_distance_per_90 = per90(carry_distance_total, minutes)
    avg_carry_length = float(length.mean()) if carries_total > 0 else 0.0

    def _stats_for_mask(mask: pd.Series, prefix: str) -> dict:
        total = int(mask.sum())
        pct = float(mask.mean()) if len(mask) else 0.0
        if total > 0:
            lens = length[mask]
            dist_total = float(lens.sum())
            dist_avg = float(lens.mean())
        else:
            dist_total = 0.0
            dist_avg = 0.0

        return {
            f"{prefix}_carries_total": total,
            f"{prefix}_carries_per_90": per90(total, minutes),
            f"{prefix}_carries_pct": pct,
            f"{prefix}_carry_distance_total": dist_total,
            f"{prefix}_carry_distance_per_90": per90(dist_total, minutes),
            f"{prefix}_carry_distance_avg": dist_avg,
        }

    result = {
        "carries_total": carries_total,
        "carries_per_90": carries_per_90,
        "carry_distance_total": carry_distance_total,
        "carry_distance_per_90": carry_distance_per_90,
        "avg_carry_length": avg_carry_length,
    }

    # directional
    result.update(_stats_for_mask(progressive_mask, "progressive"))
    result.update(_stats_for_mask(backward_mask, "backward"))

    # thirds + final third + box
    result.update(_stats_for_mask(defensive_mask, "defensive"))
    result.update(_stats_for_mask(attacking_mask, "offensive"))
    result.update(_stats_for_mask(offensive_end_mask, "final_third"))
    result.update(_stats_for_mask(box_entry_mask, "box_entry"))

    # third transitions
    result.update(_stats_for_mask(def_to_mid_mask, "def_to_mid"))
    result.update(_stats_for_mask(def_to_attack_mask, "def_to_attack"))
    result.update(_stats_for_mask(mid_to_attack_mask, "mid_to_attack"))
    result.update(_stats_for_mask(mid_to_mid_mask, "mid_to_mid"))
    result.update(_stats_for_mask(att_to_mid_mask, "att_to_mid"))
    result.update(_stats_for_mask(def_to_def_mask, "def_to_def"))
    result.update(_stats_for_mask(att_to_att_mask, "att_to_att"))

    # channel transitions
    result.update(_stats_for_mask(left_to_centre_mask,  "left_to_centre"))
    result.update(_stats_for_mask(left_to_right_mask,   "left_to_right"))
    result.update(_stats_for_mask(centre_to_left_mask,  "centre_to_left"))
    result.update(_stats_for_mask(centre_to_right_mask, "centre_to_right"))
    result.update(_stats_for_mask(right_to_centre_mask, "right_to_centre"))
    result.update(_stats_for_mask(right_to_left_mask,   "right_to_left"))

    # style masks
    result.update(_stats_for_mask(offensive_progressive_mask, "offensive_progressive"))
    result.update(_stats_for_mask(lateral_mask, "lateral"))
    result.update(_stats_for_mask(wide_lane_mask, "wide_lane"))

    # angle stats by thirds/channels of origin
    origin_third = third_from_x(start_x)
    def_third_mask = origin_third == "def"
    mid_third_mask = origin_third == "mid"
    att_third_mask = origin_third == "att"

    left_channel_mask = origin_channel == "left"
    centre_channel_mask = origin_channel == "centre"
    right_channel_mask = origin_channel == "right"

    mean_o, var_o = circular_angle_stats(theta)
    mean_d, var_d = circular_angle_stats(theta, def_third_mask)
    mean_m, var_m = circular_angle_stats(theta, mid_third_mask)
    mean_a, var_a = circular_angle_stats(theta, att_third_mask)
    mean_l, var_l = circular_angle_stats(theta, left_channel_mask)
    mean_c, var_c = circular_angle_stats(theta, centre_channel_mask)
    mean_r, var_r = circular_angle_stats(theta, right_channel_mask)

    result.update({
        "carry_angle_mean_overall": mean_o,
        "carry_angle_var_overall": var_o,
        "carry_angle_mean_def_third": mean_d,
        "carry_angle_var_def_third": var_d,
        "carry_angle_mean_mid_third": mean_m,
        "carry_angle_var_mid_third": var_m,
        "carry_angle_mean_att_third": mean_a,
        "carry_angle_var_att_third": var_a,
        "carry_angle_mean_left_channel": mean_l,
        "carry_angle_var_left_channel": var_l,
        "carry_angle_mean_centre_channel": mean_c,
        "carry_angle_var_centre_channel": var_c,
        "carry_angle_mean_right_channel": mean_r,
        "carry_angle_var_right_channel": var_r,
    })

    # direction entropy over 8 buckets
    bucket_masks = angle_bucket_masks(angle_deg)
    counts = np.array([m.sum() for m in bucket_masks.values()], dtype=float)
    tot = counts.sum()
    if tot > 0:
        p = counts / tot
        p = p[p > 0]
        carry_direction_entropy = float(-(p * np.log(p)).sum())
    else:
        carry_direction_entropy = 0.0

    max_carry_length = float(length.max()) if carries_total > 0 else 0.0
    carry_burst_index = float(result.get("progressive_carry_distance_total", 0.0)) / carry_distance_total if carry_distance_total > 0 else 0.0

    result.update({
        "carry_direction_entropy": carry_direction_entropy,
        "max_carry_length": max_carry_length,
        "carry_burst_index": carry_burst_index,
    })

    return result
