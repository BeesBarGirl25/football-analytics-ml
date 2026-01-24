from __future__ import annotations
import numpy as np
import pandas as pd

from football_analytics.config import (
    X_DEF, X_ATT, Y_LEFT, Y_CENTRE,
    Y_LEFT_WING, Y_LEFT_HALF, Y_RIGHT_HALF, Y_RIGHT_WING,
)

ZONE_KEYS_6X6 = [
    f"{xb}{yb}"
    for xb in ("d1", "d2", "m1", "m2", "a1", "a2")
    for yb in ("l1", "l2", "c1", "c2", "r1", "r2")
]


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
    """
    Hard-coded 6x6 grid based on existing third/channel boundaries:
      X: [0..X_DEF) , [X_DEF..X_ATT) , [X_ATT..max]
          split each third into two bands -> d1,d2,m1,m2,a1,a2
      Y: [0..Y_LEFT) , [Y_LEFT..Y_CENTRE) , [Y_CENTRE..max]
          split each channel into two bands -> l1,l2,c1,c2,r1,r2

    Returns 36 fractions with keys like: d1l1, d2c2, m1r1, a2l2, ...
    """
    mask = x.notna() & y.notna()
    if mask.sum() == 0:
        xb = ["d1", "d2", "m1", "m2", "a1", "a2"]
        yb = ["l1", "l2", "c1", "c2", "r1", "r2"]
        return {f"{xi}{yi}": 0.0 for xi in xb for yi in yb}

    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)

    # Use observed max as the "end" of the pitch to avoid hardcoding 120/80 here.
    # If your data is StatsBomb, this will be ~120 and ~80 anyway.
    x_max = float(np.nanmax(xv))
    y_max = float(np.nanmax(yv))

    # Avoid degenerate case where max == boundary (rare, but safe)
    x_max = max(x_max, float(X_ATT) + 1e-6)
    y_max = max(y_max, float(Y_CENTRE) + 1e-6)

    # Midpoints within each third/channel
    x_def_mid = float(X_DEF) / 2.0
    x_mid_mid = (float(X_DEF) + float(X_ATT)) / 2.0
    x_att_mid = (float(X_ATT) + x_max) / 2.0

    y_left_mid = float(Y_LEFT) / 2.0
    y_centre_mid = (float(Y_LEFT) + float(Y_CENTRE)) / 2.0
    y_right_mid = (float(Y_CENTRE) + y_max) / 2.0

    # Bin labels
    x_bins = np.select(
        [
            xv < x_def_mid,                                  # first half of def third
            (xv >= x_def_mid) & (xv < float(X_DEF)),         # second half of def third
            (xv >= float(X_DEF)) & (xv < x_mid_mid),         # first half of mid third
            (xv >= x_mid_mid) & (xv < float(X_ATT)),         # second half of mid third
            (xv >= float(X_ATT)) & (xv < x_att_mid),         # first half of att third
            xv >= x_att_mid,                                 # second half of att third
        ],
        ["d1", "d2", "m1", "m2", "a1", "a2"],
        default="m1",
    )

    y_bins = np.select(
        [
            yv < y_left_mid,
            (yv >= y_left_mid) & (yv < float(Y_LEFT)),
            (yv >= float(Y_LEFT)) & (yv < y_centre_mid),
            (yv >= y_centre_mid) & (yv < float(Y_CENTRE)),
            (yv >= float(Y_CENTRE)) & (yv < y_right_mid),
            yv >= y_right_mid,
        ],
        ["l1", "l2", "c1", "c2", "r1", "r2"],
        default="c1",
    )

    # Count fractions
    total = len(x_bins)
    out = {}
    for xb in ("d1", "d2", "m1", "m2", "a1", "a2"):
        for yb in ("l1", "l2", "c1", "c2", "r1", "r2"):
            out[f"{xb}{yb}"] = float(((x_bins == xb) & (y_bins == yb)).sum()) / float(total)

    return out


def thirds_from_zones(z: dict[str, float]) -> dict[str, float]:
    return {
        "def": sum(v for k, v in z.items() if k.startswith("d")),
        "mid": sum(v for k, v in z.items() if k.startswith("m")),
        "att": sum(v for k, v in z.items() if k.startswith("a")),
    }

def channels_from_zones(z: dict[str, float]) -> dict[str, float]:
    return {
        "left":  sum(v for k, v in z.items() if k.endswith(("l1", "l2"))),
        "centre": sum(v for k, v in z.items() if k.endswith(("c1", "c2"))),
        "right": sum(v for k, v in z.items() if k.endswith(("r1", "r2"))),
    }


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
