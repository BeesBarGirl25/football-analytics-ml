from __future__ import annotations
import numpy as np
import pandas as pd

from football_analytics.config import (
    X_DEF, X_ATT, Y_LEFT, Y_CENTRE,
    Y_LEFT_WING, Y_LEFT_HALF, Y_RIGHT_HALF, Y_RIGHT_WING,
)

def per90(value: float, minutes: float) -> float:
    return 0.0 if minutes <= 0 else float(value) * 90.0 / float(minutes)

def safe_pct(mask: pd.Series, total: int | None = None) -> float:
    if total is not None:
        return 0.0 if total <= 0 else float(mask.sum()) / float(total)
    return float(mask.mean()) if len(mask) else 0.0

def third_from_x(x: pd.Series) -> pd.Series:
    return np.select(
        [x < X_DEF, (x >= X_DEF) & (x < X_ATT), x >= X_ATT],
        ["def", "mid", "att"],
        default="mid",
    )

def channel_from_y(y: pd.Series) -> pd.Series:
    return np.select(
        [y < Y_LEFT, (y >= Y_LEFT) & (y < Y_CENTRE), y >= Y_CENTRE],
        ["left", "centre", "right"],
        default="centre",
    )

def zone_fractions(x: pd.Series, y: pd.Series) -> dict[str, float]:
    return {
        "dl": float(((x < X_DEF) & (y < Y_LEFT)).mean()),
        "dc": float(((x < X_DEF) & (y >= Y_LEFT) & (y < Y_CENTRE)).mean()),
        "dr": float(((x < X_DEF) & (y >= Y_CENTRE)).mean()),
        "ml": float(((x >= X_DEF) & (x < X_ATT) & (y < Y_LEFT)).mean()),
        "mc": float(((x >= X_DEF) & (x < X_ATT) & (y >= Y_LEFT) & (y < Y_CENTRE)).mean()),
        "mr": float(((x >= X_DEF) & (x < X_ATT) & (y >= Y_CENTRE)).mean()),
        "al": float(((x >= X_ATT) & (y < Y_LEFT)).mean()),
        "ac": float(((x >= X_ATT) & (y >= Y_LEFT) & (y < Y_CENTRE)).mean()),
        "ar": float(((x >= X_ATT) & (y >= Y_CENTRE)).mean()),
    }

def thirds_from_zones(z: dict[str, float]) -> dict[str, float]:
    return {"def": z["dl"] + z["dc"] + z["dr"],
            "mid": z["ml"] + z["mc"] + z["mr"],
            "att": z["al"] + z["ac"] + z["ar"]}

def channels_from_zones(z: dict[str, float]) -> dict[str, float]:
    return {"left": z["dl"] + z["ml"] + z["al"],
            "centre": z["dc"] + z["mc"] + z["ac"],
            "right": z["dr"] + z["mr"] + z["ar"]}

def angle_bucket_masks(angle_deg: pd.Series) -> dict[str, pd.Series]:
    return {
        "F":  (angle_deg > -22.5) & (angle_deg <= 22.5),
        "FR": (angle_deg > 22.5)  & (angle_deg <= 67.5),
        "R":  (angle_deg > 67.5)  & (angle_deg <= 112.5),
        "BR": (angle_deg > 112.5) & (angle_deg <= 157.5),
        "B":  (angle_deg > 157.5) | (angle_deg <= -157.5),
        "BL": (angle_deg > -157.5) & (angle_deg <= -112.5),
        "L":  (angle_deg > -112.5) & (angle_deg <= -67.5),
        "FL": (angle_deg > -67.5)  & (angle_deg <= -22.5),
    }

def circular_angle_stats(theta: pd.Series, mask: pd.Series | None = None) -> tuple[float, float]:
    if mask is not None:
        theta = theta[mask]
    theta = theta.dropna()
    if theta.empty:
        return 0.0, 0.0

    sin_mean = np.sin(theta).mean()
    cos_mean = np.cos(theta).mean()

    mean_angle = float(np.degrees(np.arctan2(sin_mean, cos_mean)))
    R = float(np.sqrt(sin_mean**2 + cos_mean**2))
    var = 1.0 - R
    return mean_angle, var
