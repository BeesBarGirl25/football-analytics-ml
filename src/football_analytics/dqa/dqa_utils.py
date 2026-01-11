from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from football_analytics.dqa.positional_diagnostics import (
    position_centroid_similarity,
    within_position_variance,
    position_separability,
    feature_position_signal,
)
# -----------------------------
# Helpers
# -----------------------------

def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric-only frame (excluding boolean by default is fine; pandas treats bool as number sometimes)."""
    num = df.select_dtypes(include=["number"]).copy()
    # Convert inf to nan so downstream stats behave
    num.replace([np.inf, -np.inf], np.nan, inplace=True)
    return num


def _safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except Exception:
        return 0


def _is_probably_ratio_col(col: str) -> bool:
    c = col.lower()
    return c.endswith(("_pct", "_rate", "_share", "_ratio", "_prop", "_fraction"))


# -----------------------------
# Core DQA
# -----------------------------

def column_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Structural overview per column:
    - dtype
    - non-null / null counts + %
    - unique count
    - constant flag
    - inf count (for numeric)
    """
    nulls = df.isna().sum()
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "rows": len(df),
            "non_null": df.count(),
            "nulls": nulls,
            "null_%": (nulls / max(len(df), 1) * 100).round(2),
            "unique": [ _safe_nunique(df[c]) for c in df.columns ],
            "constant": [ _safe_nunique(df[c]) <= 1 for c in df.columns ],
        },
        index=df.columns,
    )

    # inf count (numeric only)
    num = df.select_dtypes(include=["number"])
    if not num.empty:
        inf_count = np.isinf(num.to_numpy()).sum(axis=0)
        summary.loc[num.columns, "inf"] = inf_count
    else:
        summary["inf"] = 0

    summary["inf"] = summary["inf"].fillna(0).astype(int)
    return summary.sort_values(["null_%", "constant", "unique"], ascending=[False, False, True])


def numeric_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spread stats for numeric columns:
    - count/mean/std/min/25/50/75/max
    - IQR
    - zero_var
    - near_zero_var (very small variance)
    """
    num = _numeric_df(df)
    if num.empty:
        return pd.DataFrame()

    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["iqr"] = desc["75%"] - desc["25%"]
    desc["zero_var"] = desc["std"].fillna(0) == 0

    # Heuristic: near-zero variance relative to scale
    # If std is tiny vs |mean| or range, flag it.
    rng = (desc["max"] - desc["min"]).replace(0, np.nan)
    scale = (desc["mean"].abs() + rng).replace(0, np.nan)
    rel_std = (desc["std"] / scale).fillna(0)
    desc["near_zero_var"] = rel_std < 1e-6

    # Convenience columns
    desc["missing_%"] = (1 - (desc["count"] / max(len(df), 1))) * 100
    desc["missing_%"] = desc["missing_%"].round(2)

    return desc.sort_values(["zero_var", "near_zero_var", "iqr"], ascending=[False, False, True])


def outlier_pressure(df: pd.DataFrame, method: str = "iqr", iqr_k: float = 1.5) -> pd.Series:
    """
    Fraction of rows flagged as outliers per numeric feature.

    method:
      - "iqr" : Tukey fences using Q1/Q3 +/- k*IQR
      - "z"   : |z| > 3 (robust-ish if near-normal; less good on heavy tails)

    Returns: Series indexed by column name with outlier fraction in [0,1].
    """
    num = _numeric_df(df)
    if num.empty or len(df) == 0:
        return pd.Series(dtype=float)

    if method.lower() == "iqr":
        q1 = num.quantile(0.25)
        q3 = num.quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        lower = q1 - iqr_k * iqr
        upper = q3 + iqr_k * iqr
        mask = (num.lt(lower)) | (num.gt(upper))
        frac = mask.sum(axis=0) / len(df)
        return frac.sort_values(ascending=False)

    if method.lower() == "z":
        mu = num.mean(axis=0)
        sd = num.std(axis=0).replace(0, np.nan)
        z = (num - mu) / sd
        frac = (z.abs() > 3).sum(axis=0) / len(df)
        return frac.sort_values(ascending=False)

    raise ValueError("method must be one of {'iqr','z'}")


def range_violations(df: pd.DataFrame, cols: Sequence[str] | None = None) -> pd.Series:
    """
    For columns that represent ratios/rates/shares, flag fraction of rows outside [0,1].
    Default behaviour: auto-detect columns by suffix: _pct/_rate/_share/_ratio/_prop/_fraction
    """
    num = _numeric_df(df)
    if num.empty or len(df) == 0:
        return pd.Series(dtype=float)

    if cols is None:
        cols = [c for c in num.columns if _is_probably_ratio_col(c)]
    cols = [c for c in cols if c in num.columns]

    if not cols:
        return pd.Series(dtype=float)

    bad = {}
    for c in cols:
        s = num[c]
        bad[c] = float(((s < 0) | (s > 1)).mean())
    return pd.Series(bad).sort_values(ascending=False)


# -----------------------------
# Semantic / Integrity checks
# -----------------------------

def bucket_collapse(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    suffix: str = "_share",
    max_unique: int = 3,
    min_non_null_frac: float = 0.2,
) -> pd.Series:
    """
    Detect "bucket collapse" for discretised/bucketed share columns (zones, channels, angles).

    A column is flagged True if:
      - it has <= max_unique distinct (non-null) values (default <=3), AND
      - it has enough data (non-null fraction >= min_non_null_frac)

    This catches broken bucketing that yields only {0, 0.5, 1} or similar.
    """
    if cols is None:
        cols = [c for c in df.columns if c.endswith(suffix)]

    out = {}
    n = max(len(df), 1)

    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        non_null_frac = float(s.notna().mean())
        nun = _safe_nunique(s)
        out[c] = (non_null_frac >= min_non_null_frac) and (nun <= max_unique)

    return pd.Series(out).sort_values(ascending=False)


def correlated_features(
    df: pd.DataFrame,
    thresh: float = 0.995,
    method: str = "pearson",
    fillna: float | None = 0.0,
    sample: int | None = None,
) -> pd.Series:
    """
    Find highly correlated feature pairs (redundant / duplicates).

    Returns a Series indexed by (colA, colB) tuples with abs(corr) value.

    Notes:
      - Works best on the final merged feature table.
      - fillna defaults to 0.0 to allow correlation computation; set to None to drop rows pairwise.
      - sample can be used to subsample rows for speed on huge tables.
    """
    num = _numeric_df(df)
    if num.empty:
        return pd.Series(dtype=float)

    if sample is not None and len(num) > sample:
        num = num.sample(sample, random_state=42)

    if fillna is not None:
        num = num.fillna(fillna)

    corr = num.corr(method=method).abs()
    # upper triangle without diagonal
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().sort_values(ascending=False)
    return pairs[pairs >= thresh]


def sparse_features(
    df: pd.DataFrame,
    thresh: float = 0.95,
    zeros: Iterable[float] = (0.0,),
) -> pd.Series:
    """
    Identify numeric features that are mostly zero (or mostly in a small 'zero-ish' set).

    Returns: Series of zero-ish fraction per column (descending), filtered to >= thresh.
    """
    num = _numeric_df(df)
    if num.empty or len(df) == 0:
        return pd.Series(dtype=float)

    zeros_set = set(zeros)

    frac = {}
    for c in num.columns:
        s = num[c]
        frac[c] = float(s.isin(zeros_set).mean())
    frac = pd.Series(frac).sort_values(ascending=False)
    return frac[frac >= thresh]


def low_entropy(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    bins: int = 10,
    normalize: bool = True,
) -> pd.Series:
    """
    Compute (Shannon) entropy of numeric columns using histogram binning.
    Low entropy => low information content / near-constant / collapsed distribution.

    - If normalize=True, returns entropy / log(bins) in [0,1] (when bins>1).
    """
    num = _numeric_df(df)
    if num.empty:
        return pd.Series(dtype=float)

    if cols is None:
        cols = list(num.columns)
    cols = [c for c in cols if c in num.columns]

    ent = {}
    for c in cols:
        x = num[c].dropna().to_numpy()
        if x.size == 0:
            ent[c] = np.nan
            continue
        # If constant, entropy 0
        if np.all(x == x[0]):
            ent[c] = 0.0
            continue

        hist, _ = np.histogram(x, bins=bins)
        p = hist.astype(float)
        p_sum = p.sum()
        if p_sum == 0:
            ent[c] = np.nan
            continue
        p = p / p_sum
        p = p[p > 0]
        h = float(-(p * np.log(p)).sum())
        if normalize and bins > 1:
            h = h / float(np.log(bins))
        ent[c] = h

    return pd.Series(ent).sort_values(ascending=True)


# -----------------------------
# Leakage / Rules
# -----------------------------

def identity_leakage(
    df: pd.DataFrame,
    group_col: str = "player",
    cols: Sequence[str] | None = None,
    max_groups: int | None = None,
) -> pd.Series:
    """
    Heuristic leakage detector:
    For each numeric feature, compute the maximum number of unique values within any single identity group.

    Intuition:
      - If within a player, the feature takes many distinct values across rows, that’s not leakage.
      - Leakage risk is more about features that strongly encode identity across the dataset,
        but a simple, actionable heuristic is:
           "features that are effectively constant per player and distinct between players"
        We approximate this by computing per-player unique counts and summarising.

    Output includes:
      - per_feature: fraction of players for which feature is constant (nunique==1)
    """
    if group_col not in df.columns:
        return pd.Series(dtype=float)

    num = _numeric_df(df)
    if num.empty:
        return pd.Series(dtype=float)

    if cols is None:
        cols = list(num.columns)
    cols = [c for c in cols if c in num.columns]

    g = df[[group_col]].join(num[cols])

    # Optional cap for speed
    if max_groups is not None:
        # Keep most frequent groups
        top_groups = g[group_col].value_counts().head(max_groups).index
        g = g[g[group_col].isin(top_groups)]

    # For each feature, fraction of groups where the feature is constant
    const_frac = {}
    grp = g.groupby(group_col, observed=True)
    for c in cols:
        nun_per_group = grp[c].nunique(dropna=True)
        const_frac[c] = float((nun_per_group <= 1).mean())

    return pd.Series(const_frac).sort_values(ascending=False)


def impossible_states(
    df: pd.DataFrame,
    rules: Sequence[tuple[str, str]] | None = None,
) -> pd.Series:
    """
    Generic "impossible state" checker for count columns.
    rules is a list of (a, b) meaning 'a should never exceed b' (a <= b).

    Returns: Series mapping 'a_gt_b' -> fraction of rows where violated.

    Example rules:
        [
          ("pass_completed", "pass_attempted"),
          ("forward_passes", "passes"),
          ("long_passes", "passes"),
        ]
    """
    if rules is None:
        rules = []

    out = {}
    n = max(len(df), 1)

    for a, b in rules:
        if a in df.columns and b in df.columns:
            va = df[a]
            vb = df[b]
            out[f"{a}_gt_{b}"] = float((va > vb).sum() / n)

    return pd.Series(out).sort_values(ascending=False)


# -----------------------------
# One-call report
# -----------------------------

@dataclass(frozen=True)
class DQAReport:
    health: pd.DataFrame
    spread: pd.DataFrame
    outliers: pd.Series
    range_viol: pd.Series
    bucket_collapsed: pd.Series
    sparse: pd.Series
    low_entropy: pd.Series
    correlated: pd.Series
    leakage: pd.Series
    impossible: pd.Series
    extras: dict = field(default_factory=dict)


def feature_quality_report(
    df: pd.DataFrame,
    *,
    group_col: str = "player",
    ratio_cols: Sequence[str] | None = None,
    bucket_suffix: str = "_share",
    impossible_rules: Sequence[tuple[str, str]] | None = None,
    corr_thresh: float = 0.995,
    sparse_thresh: float = 0.95,
    entropy_bins: int = 10,
) -> DQAReport:
    """
    Run a standard suite of DQA checks and return a structured report.
    This is designed to be easy to log/inspect in notebooks or pipelines.
    """
    h = column_health(df)
    sp = numeric_spread(df)
    out = outlier_pressure(df, method="iqr")
    rv = range_violations(df, cols=ratio_cols)
    bc = bucket_collapse(df, suffix=bucket_suffix)
    sf = sparse_features(df, thresh=sparse_thresh)
    le = low_entropy(df, bins=entropy_bins)
    corr = correlated_features(df, thresh=corr_thresh)
    leak = identity_leakage(df, group_col=group_col)
    imp = impossible_states(df, rules=impossible_rules or [])
    
    extras = {}
    if "player_position" in df.columns:
        extras["position_centroid_similarity"] = position_centroid_similarity(df)
        extras["within_position_variance"] = within_position_variance(df)
        extras["position_separability"] = position_separability(df)
        extras["feature_position_signal"] = feature_position_signal(df)

    return DQAReport(
        health=h,
        spread=sp,
        outliers=out,
        range_viol=rv,
        bucket_collapsed=bc,
        sparse=sf,
        low_entropy=le,
        correlated=corr,
        leakage=leak,
        impossible=imp,
        extras = extras
    )


__all__ = [
    "DQAReport",
    "column_health",
    "numeric_spread",
    "outlier_pressure",
    "range_violations",
    "bucket_collapse",
    "correlated_features",
    "identity_leakage",
    "sparse_features",
    "low_entropy",
    "impossible_states",
    "feature_quality_report",
]