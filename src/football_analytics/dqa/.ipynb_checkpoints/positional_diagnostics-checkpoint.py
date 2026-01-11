"""
football_analytics.dqa.positional_diagnostics

Diagnostics designed for *one-row-per-player* feature tables.

Context:
- When your dataset is aggregated to one row per player (e.g., per season/tournament),
  "identity leakage" checks become largely meaningless because each row is literally a
  player fingerprint by design.

Instead, these functions answer the more useful questions:
- Are positions/archetypes coherent (low within-position variance)?
- Are positions separable in your feature space (do roles cluster)?
- Which positions overlap most (e.g., FB vs WB vs wide CM)?
- Which features actually differentiate positions?

All functions are intended to be integrated into DQAReport.extras (Option A).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric-only dataframe with inf coerced to NaN."""
    num = df.select_dtypes(include=["number"]).copy()
    num.replace([np.inf, -np.inf], np.nan, inplace=True)
    return num


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise numeric columns to mean=0, std=1.
    - Fills NaNs with 0 after standardisation so downstream distance metrics behave.
    """
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0, np.nan)
    z = (df - mu) / sd
    return z.fillna(0.0)


def _safe_positions(df: pd.DataFrame, pos_col: str) -> pd.Series:
    """Return positions as string series (handles missing col gracefully)."""
    if pos_col not in df.columns:
        raise KeyError(f"pos_col='{pos_col}' not found in df columns")
    return df[pos_col].astype(str)


# -----------------------------
# Core positional diagnostics
# -----------------------------

def position_centroids(
    df: pd.DataFrame,
    pos_col: str = "player_position",
    cols: Sequence[str] | None = None,
    standardise: bool = True,
    min_count: int = 5,
) -> pd.DataFrame:
    """
    Compute per-position feature centroids.

    Why:
    - This is the core building block for position overlap analyses.
    - If you cluster or compare roles, centroids are the simplest archetype summary.

    Args:
        df: Feature table (one row per player).
        pos_col: Column containing position labels.
        cols: Optional subset of numeric feature columns to use. If None, uses all numeric.
        standardise: If True, z-score features before averaging so scale doesn't dominate.
        min_count: Minimum players required to include a position.

    Returns:
        DataFrame indexed by position with centroid values for each feature.
    """
    pos = _safe_positions(df, pos_col)
    num = _numeric_df(df)

    if cols is not None:
        cols = [c for c in cols if c in num.columns]
        num = num[cols]

    if standardise:
        num = _zscore(num.fillna(0.0))
    else:
        num = num.fillna(0.0)

    tmp = num.copy()
    tmp[pos_col] = pos.values

    counts = tmp[pos_col].value_counts()
    keep = counts[counts >= min_count].index
    tmp = tmp[tmp[pos_col].isin(keep)]

    return tmp.groupby(pos_col, observed=True).mean()


def position_centroid_similarity(
    df: pd.DataFrame,
    pos_col: str = "player_position",
    cols: Sequence[str] | None = None,
    metric: str = "cosine",
    standardise: bool = True,
    min_count: int = 5,
) -> pd.DataFrame:
    """
    Compute similarity matrix between position centroids.

    Why:
    - Answers: "Which positions look the most similar in this feature space?"
    - Great for validating whether your features reflect football intuition
      (e.g., CB close to DM? WB close to Winger? etc.)

    Args:
        df: Feature table (one row per player).
        pos_col: Column containing position labels.
        cols: Optional subset of numeric features.
        metric: "cosine" (recommended) or "euclidean" (converted to similarity).
        standardise: Z-score features before centroid calculation (recommended).
        min_count: Minimum players required to include a position.

    Returns:
        Square DataFrame (positions × positions) with similarity scores.
        - For cosine: range [-1, 1] (typically [0,1] if features are mostly aligned).
        - For euclidean: converted to similarity via 1/(1+d), range (0,1].
    """
    cent = position_centroids(
        df, pos_col=pos_col, cols=cols, standardise=standardise, min_count=min_count
    )

    if cent.empty:
        return pd.DataFrame()

    X = cent.to_numpy(dtype=float)
    labels = cent.index.astype(str).tolist()

    if metric == "cosine":
        # Cosine similarity: (A·B)/(|A||B|)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        sim = Xn @ Xn.T
        return pd.DataFrame(sim, index=labels, columns=labels)

    if metric == "euclidean":
        # Pairwise distances then convert to similarity
        # similarity = 1 / (1 + dist)
        dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        sim = 1.0 / (1.0 + dists)
        return pd.DataFrame(sim, index=labels, columns=labels)

    raise ValueError("metric must be one of {'cosine','euclidean'}")


def within_position_variance(
    df: pd.DataFrame,
    pos_col: str = "player_position",
    cols: Sequence[str] | None = None,
    standardise: bool = True,
    min_count: int = 5,
    agg: str = "mean",
) -> pd.Series:
    """
    Measure how internally consistent each position is.

    Why:
    - Answers: "Are players in the same position behaving similarly?"
    - High within-position variance often indicates:
        * position labels are too broad (e.g., "Midfielder")
        * mixed roles inside the group (e.g., DM vs 8 vs 10)
        * your feature set isn't capturing role structure well

    Method:
    - Standardise features (optional)
    - Compute per-position variance for each feature
    - Aggregate variances to a single score per position (mean/median)

    Args:
        df: Feature table (one row per player).
        pos_col: Position label column.
        cols: Optional subset of numeric features.
        standardise: If True, z-score features first (recommended for comparability).
        min_count: Minimum players required to include a position.
        agg: "mean" or "median" aggregation over features.

    Returns:
        Series indexed by position, where larger values mean "more diverse within position".
    """
    pos = _safe_positions(df, pos_col)
    num = _numeric_df(df)

    if cols is not None:
        cols = [c for c in cols if c in num.columns]
        num = num[cols]

    if standardise:
        num = _zscore(num.fillna(0.0))
    else:
        num = num.fillna(0.0)

    tmp = num.copy()
    tmp[pos_col] = pos.values

    counts = tmp[pos_col].value_counts()
    keep = counts[counts >= min_count].index
    tmp = tmp[tmp[pos_col].isin(keep)]

    var_by_pos = tmp.groupby(pos_col, observed=True).var(ddof=0)  # population var
    if agg == "mean":
        score = var_by_pos.mean(axis=1)
    elif agg == "median":
        score = var_by_pos.median(axis=1)
    else:
        raise ValueError("agg must be one of {'mean','median'}")

    return score.sort_values(ascending=True)


def position_separability(
    df: pd.DataFrame,
    pos_col: str = "player_position",
    cols: Sequence[str] | None = None,
    standardise: bool = True,
    min_count: int = 5,
    sample: int | None = None,
) -> float:
    """
    Quantify how separable positions are in your feature space using silhouette score.

    Why:
    - Answers: "Do positions form clusters in this feature space?"
    - Useful to track over time as you add new feature families.

    Notes:
    - Silhouette score is in [-1, 1]
      * ~0 means overlapping clusters
      * >0.2 often indicates meaningful separation (rough heuristic)
      * negative means many points are closer to other clusters than their own

    Args:
        df: Feature table (one row per player).
        pos_col: Position label column.
        cols: Optional subset of numeric features.
        standardise: Z-score features (recommended).
        min_count: Minimum players required to include a position.
        sample: Optional random subsample size for speed.

    Returns:
        float silhouette score.

    Requirements:
        scikit-learn must be installed.
    """
    from sklearn.metrics import silhouette_score

    pos = _safe_positions(df, pos_col)
    num = _numeric_df(df)

    if cols is not None:
        cols = [c for c in cols if c in num.columns]
        num = num[cols]

    tmp = num.copy()
    tmp[pos_col] = pos.values

    counts = tmp[pos_col].value_counts()
    keep = counts[counts >= min_count].index
    tmp = tmp[tmp[pos_col].isin(keep)]

    if tmp.empty or tmp[pos_col].nunique() < 2:
        return float("nan")

    if sample is not None and len(tmp) > sample:
        tmp = tmp.sample(sample, random_state=42)

    X = tmp.drop(columns=[pos_col])
    if standardise:
        X = _zscore(X.fillna(0.0))
    else:
        X = X.fillna(0.0)

    y = tmp[pos_col].astype("category").cat.codes
    return float(silhouette_score(X, y))


def feature_position_signal(
    df: pd.DataFrame,
    pos_col: str = "player_position",
    cols: Sequence[str] | None = None,
    standardise: bool = True,
    min_count: int = 5,
) -> pd.Series:
    """
    Rank features by how much they differentiate positions (ANOVA-style effect size).

    Why:
    - Answers: "Which features actually separate roles?"
    - Helps you validate feature families and prune weak/noisy features.

    Method:
    - For each feature:
        between-group variance / total variance
      This yields an effect-size-like score in [0, 1], where higher means:
        "a lot of variance is explained by position differences".

    Args:
        df: Feature table (one row per player).
        pos_col: Position label column.
        cols: Optional subset of numeric features.
        standardise: Z-score features first (recommended for comparability).
        min_count: Minimum players required to include a position.

    Returns:
        Series of feature scores (descending), higher = more positional signal.
    """
    pos = _safe_positions(df, pos_col)
    num = _numeric_df(df)

    if cols is not None:
        cols = [c for c in cols if c in num.columns]
        num = num[cols]

    if standardise:
        num = _zscore(num.fillna(0.0))
    else:
        num = num.fillna(0.0)

    tmp = num.copy()
    tmp[pos_col] = pos.values

    counts = tmp[pos_col].value_counts()
    keep = counts[counts >= min_count].index
    tmp = tmp[tmp[pos_col].isin(keep)]

    if tmp.empty:
        return pd.Series(dtype=float)

    overall_mean = tmp.drop(columns=[pos_col]).mean(axis=0)

    scores = {}
    total_var = tmp.drop(columns=[pos_col]).var(ddof=0)

    # Between-group variance: sum_n (mean_g - mean_total)^2 * (n_g / n_total)
    n_total = len(tmp)
    for feat in overall_mean.index:
        if total_var[feat] == 0 or np.isnan(total_var[feat]):
            scores[feat] = 0.0
            continue

        by_pos = tmp.groupby(pos_col, observed=True)[feat].agg(["mean", "count"])
        between = ((by_pos["mean"] - overall_mean[feat]) ** 2 * (by_pos["count"] / n_total)).sum()
        scores[feat] = float(between / total_var[feat])

    return pd.Series(scores).sort_values(ascending=False)


__all__ = [
    "position_centroids",
    "position_centroid_similarity",
    "within_position_variance",
    "position_separability",
    "feature_position_signal",
]
