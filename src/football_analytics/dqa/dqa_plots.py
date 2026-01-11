from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from football_analytics.dqa.dqa_utils import DQAReport
from football_analytics.dqa.dqa_style import (
    set_dqa_theme,
    apply_dqa_ax_style,
    dqa_accent,
    DQA_COLORS,  # optional if you want warn/bad colors; remove if not exported
)

set_dqa_theme(font_scale=1.0)
PlotBundle = Dict[str, object]


# -----------------------------
# Plots
# -----------------------------

def plot_ranked_bar(
    s: pd.Series,
    *,
    title: str,
    top_n: int = 20,
    xlabel: str = "",
    ascending: bool = False,
) -> plt.Figure:
    """
    Ranked horizontal bar plot.
    s: Series indexed by feature name.
    """
    s = s.dropna()

    if s.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    s = s.sort_values(ascending=ascending).head(top_n)

    fig_h = max(4, 0.35 * len(s) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ax.barh(s.index.astype(str), s.values, color=dqa_accent())
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    apply_dqa_ax_style(ax)
    fig.tight_layout(pad=1.2)
    return fig


def plot_heatmap(
    mat: pd.DataFrame,
    *,
    title: str,
    vmax: float = 1.0,
    vmin: float = 0.0,
) -> plt.Figure:
    """
    Heatmap plot helper.
    """
    if mat is None or mat.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    # Size scales with matrix dimension
    n = max(mat.shape)
    size = max(6, min(14, 0.35 * n))
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
        mat,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        square=True,
        cbar=True,
    )

    ax.set_title(title)
    fig.tight_layout(pad=1.2)
    return fig


# -----------------------------
# Data Gathering
# -----------------------------

def get_missingness_series(report: DQAReport) -> pd.Series:
    if report.health is None or report.health.empty:
        return pd.Series(dtype=float)
    if "null_%" not in report.health.columns:
        return pd.Series(dtype=float)
    return report.health["null_%"]


def get_outlier_pressure_series(report: DQAReport) -> pd.Series:
    if report.outliers is None:
        return pd.Series(dtype=float)
    s = report.outliers
    if isinstance(s, pd.Series):
        return s
    try:
        return pd.Series(s)
    except Exception:
        return pd.Series(dtype=float)


def get_range_violations_series(report: DQAReport) -> pd.Series:
    if report.range_viol is None:
        return pd.Series(dtype=float)
    s = report.range_viol
    if isinstance(s, pd.Series):
        return s
    try:
        return pd.Series(s)
    except Exception:
        return pd.Series(dtype=float)


def get_sparse_features_series(report: DQAReport) -> pd.Series:
    if report.sparse is None:
        return pd.Series(dtype=float)
    s = report.sparse
    if isinstance(s, pd.Series):
        return s
    try:
        return pd.Series(s)
    except Exception:
        return pd.Series(dtype=float)


def get_low_entropy_series(report: DQAReport) -> pd.Series:
    if report.low_entropy is None:
        return pd.Series(dtype=float)
    s = report.low_entropy
    if isinstance(s, pd.Series):
        return s
    try:
        return pd.Series(s)
    except Exception:
        return pd.Series(dtype=float)


def get_impossible_states_series(report: DQAReport) -> pd.Series:
    if report.impossible is None:
        return pd.Series(dtype=float)
    s = report.impossible
    if isinstance(s, pd.Series):
        return s
    try:
        return pd.Series(s)
    except Exception:
        return pd.Series(dtype=float)


def get_correlated_pairs_series(report: DQAReport) -> pd.Series:
    if report.correlated is None:
        return pd.Series(dtype=float)
    s = report.correlated
    if isinstance(s, pd.Series):
        return s
    try:
        return pd.Series(s)
    except Exception:
        return pd.Series(dtype=float)


def get_extra(report: DQAReport, key: str) -> Any:
    if report.extras is None:
        return None
    return report.extras.get(key)


def get_position_signal_series(report: DQAReport) -> pd.Series:
    x = get_extra(report, "feature_position_signal")
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x
    try:
        return pd.Series(x)
    except Exception:
        return pd.Series(dtype=float)


def get_within_position_variance(report: DQAReport) -> Any:
    # Can be Series or DataFrame depending on your implementation.
    return get_extra(report, "within_position_variance")


def get_position_separability(report: DQAReport) -> Any:
    # Expecting a DataFrame-like matrix
    return get_extra(report, "position_separability")


# -----------------------------
# Registry-driven render
# -----------------------------

def render_plot(
    key: str,
    df: pd.DataFrame,
    report: DQAReport,
) -> Optional[Dict[str, object]]:
    if key not in PLOT_REGISTRY:
        raise KeyError(f"Unknown plot key: {key}")

    spec = PLOT_REGISTRY[key]
    cfg = dict(spec.get("config", {}))

    title = spec.get("title", key)
    description = spec.get("description", "")

    cfg["title"] = title

    data = spec["get_data"](df, report)

    min_value = cfg.get("min_value", None)
    if min_value is not None and isinstance(data, pd.Series):
        if data.empty or float(data.max()) < float(min_value):
            return None

    fig = spec["plot"](data, df, report, cfg)
    plt.close(fig) 

    return {
        "key": key,
        "title": title,
        "description": description,
        "fig": fig,
    }



def render_dashboard(
    df: pd.DataFrame,
    report: DQAReport,
    keys: Optional[list[str]] = None,
) -> Dict[str, Dict[str, object]]:
    if keys is None:
        keys = [
            "missingness_hotspots",
            "outlier_pressure",
            "range_violations",
            "sparse_features",
            "low_entropy",
            "impossible_states",
            "correlation_cluster",
            "position_signal",
            "position_separability",
            "within_position_variance",
        ]

    plots: Dict[str, Dict[str, object]] = {}
    for k in keys:
        bundle = render_plot(k, df, report)
        if bundle is not None:
            plots[k] = bundle
    return plots



# -----------------------------
# Plot builders used by registry
# -----------------------------

def _plot_ranked_bar_from_series(data: pd.Series, *, title: str, top_n: int, xlabel: str, ascending: bool = False) -> plt.Figure:
    return plot_ranked_bar(
        data,
        title=title,
        top_n=top_n,
        xlabel=xlabel,
        ascending=ascending,
    )


def _plot_correlation_cluster(
    pairs: pd.Series,
    df: pd.DataFrame,
    *,
    title: str,
    top_k_pairs: int = 50,
    max_cols: int = 30,
    sample_rows: Optional[int] = 5000,
    fillna: float = 0.0,
) -> plt.Figure:
    """
    Build a correlation heatmap from the top-K correlated pairs (subset),
    to avoid huge unreadable full-matrix plots.
    """
    pairs = pairs.dropna()
    if pairs.empty:
        return plot_heatmap(pd.DataFrame(), title=title)

    top_pairs = pairs.sort_values(ascending=False).head(top_k_pairs)

    # Pairs index is expected to be (colA, colB)
    cols = []
    for idx in top_pairs.index:
        try:
            a, b = idx
            cols.extend([a, b])
        except Exception:
            continue

    if not cols:
        return plot_heatmap(pd.DataFrame(), title=title)

    # Keep most frequent columns if too many
    col_counts = pd.Series(cols).value_counts()
    cols = list(col_counts.head(max_cols).index)

    num = df.select_dtypes(include=["number"]).copy()
    num.replace([np.inf, -np.inf], np.nan, inplace=True)

    cols = [c for c in cols if c in num.columns]
    if len(cols) < 2:
        return plot_heatmap(pd.DataFrame(), title=title)

    sub = num[cols]
    if sample_rows is not None and len(sub) > sample_rows:
        sub = sub.sample(sample_rows, random_state=42)

    if fillna is not None:
        sub = sub.fillna(fillna)

    corr = sub.corr().abs()
    return plot_heatmap(corr, title=title, vmin=0.0, vmax=1.0)


def _plot_position_separability(mat: Any, *, title: str) -> plt.Figure:
    # Accept DataFrame or ndarray-like
    if mat is None:
        return plot_heatmap(pd.DataFrame(), title=title)

    if isinstance(mat, pd.DataFrame):
        df_mat = mat
    else:
        try:
            df_mat = pd.DataFrame(mat)
        except Exception:
            df_mat = pd.DataFrame()

    return plot_heatmap(df_mat, title=title, vmin=0.0, vmax=float(np.nanmax(df_mat.to_numpy())) if not df_mat.empty else 1.0)


def _plot_within_position_variance(x: Any, *, title: str, top_n: int = 20) -> plt.Figure:
    """
    If you return a Series -> ranked bar.
    If you return a DataFrame -> try to summarise into a Series (mean) and plot that.
    """
    if x is None:
        return plot_ranked_bar(pd.Series(dtype=float), title=title, top_n=top_n)

    if isinstance(x, pd.Series):
        return plot_ranked_bar(x, title=title, top_n=top_n, xlabel="Variance")

    if isinstance(x, pd.DataFrame) and not x.empty:
        # Try a sensible default: mean variance across positions
        numeric = x.select_dtypes(include=["number"])
        if not numeric.empty:
            s = numeric.mean(axis=0).sort_values(ascending=False)
            return plot_ranked_bar(s, title=title, top_n=top_n, xlabel="Mean variance")
        return plot_ranked_bar(pd.Series(dtype=float), title=title, top_n=top_n)

    return plot_ranked_bar(pd.Series(dtype=float), title=title, top_n=top_n)


# -----------------------------
# Registry
# -----------------------------

PLOT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "missingness_hotspots": {
        "title": "Missingness hotspots (null %)",
        "description": "Features with the highest proportion of missing values. Highlights broken joins and dropped player segments.",
        "get_data": lambda df, report: get_missingness_series(report),
        "plot": lambda data, df, report, cfg: _plot_ranked_bar_from_series(
            data,
            title=cfg["title"],
            top_n=cfg["top_n"],
            xlabel="Null (%)",
        ),
        "config": {"top_n": 20, "min_value": 0.0},
    },

    "outlier_pressure": {
        "title": "Outlier pressure (IQR, % rows flagged)",
        "description": "Fraction of rows flagged as distribution outliers per feature. Catches broken scaling and unstable rate features.",
        "get_data": lambda df, report: get_outlier_pressure_series(report) * 100.0,
        "plot": lambda data, df, report, cfg: _plot_ranked_bar_from_series(
            data,
            title=cfg["title"],
            top_n=cfg["top_n"],
            xlabel="Outlier rows (%)",
        ),
        "config": {"top_n": 20, "min_value": 0.5},
    },

    "range_violations": {
        "title": "Ratio range violations",
        "description": "Fraction of rows where ratio features fall outside [0,1]. Detects percent encoding and normalisation bugs.",
        "get_data": lambda df, report: get_range_violations_series(report) * 100.0,
        "plot": lambda data, df, report, cfg: _plot_ranked_bar_from_series(
            data,
            title=cfg["title"],
            top_n=cfg["top_n"],
            xlabel="Violations (%)",
        ),
        "config": {"top_n": 20, "min_value": 0.1},
    },

    "sparse_features": {
        "title": "Sparse features (mostly zero)",
        "description": "Features that are mostly zero. Identifies dead or low-signal features.",
        "get_data": lambda df, report: get_sparse_features_series(report) * 100.0,
        "plot": lambda data, df, report, cfg: _plot_ranked_bar_from_series(
            data,
            title=cfg["title"],
            top_n=cfg["top_n"],
            xlabel="Zero-ish rows (%)",
        ),
        "config": {"top_n": 20, "min_value": 90.0},  # only show if something is extremely sparse
    },

    "low_entropy": {
        "title": "Low entropy features",
        "description": "Features with collapsed or near-constant distributions. Detects broken bucketing and low information content.",
        "get_data": lambda df, report: get_low_entropy_series(report),
        "plot": lambda data, df, report, cfg: plot_ranked_bar(
            data,
            title=cfg["title"],
            top_n=cfg["top_n"],
            xlabel="Entropy (lower = worse)",
            ascending=True,  # show lowest first
        ),
        "config": {"top_n": 20, "min_value": None},
    },

    "impossible_states": {
        "title": "Impossible state violations",
        "description": "Logical contradictions between count features (e.g. completed > attempted). Detects pipeline logic bugs.",
        "get_data": lambda df, report: get_impossible_states_series(report) * 100.0,
        "plot": lambda data, df, report, cfg: _plot_ranked_bar_from_series(
            data,
            title=cfg["title"],
            top_n=cfg["top_n"],
            xlabel="Rows violated (%)",
        ),
        "config": {"top_n": 20, "min_value": 0.1},
    },

    "correlation_cluster": {
        "title": "Correlation cluster (subset heatmap)",
        "description": "Groups of highly correlated features. Highlights duplicate or redundant joins.",
        "get_data": lambda df, report: get_correlated_pairs_series(report),
        "plot": lambda data, df, report, cfg: _plot_correlation_cluster(
            data,
            df,
            title=cfg["title"],
            top_k_pairs=cfg["top_k_pairs"],
            max_cols=cfg["max_cols"],
            sample_rows=cfg["sample_rows"],
            fillna=cfg["fillna"],
        ),
        "config": {"top_k_pairs": 50, "max_cols": 30, "sample_rows": 5000, "fillna": 0.0, "min_value": None},
    },

    # ----- Extras (auto-skip if missing) -----

    "position_signal": {
        "title": "Feature position signal",
        "description": "Features that best separate player positions. Identifies semantically meaningful signals.",
        "get_data": lambda df, report: get_position_signal_series(report),
        "plot": lambda data, df, report, cfg: _plot_ranked_bar_from_series(
            data,
            title=cfg["title"],
            top_n=cfg["top_n"],
            xlabel="Signal (higher = stronger)",
        ),
        "config": {"top_n": 20, "min_value": None},
    },

    "position_separability": {
        "title": "Position separability",
        "description": "How well player positions are separated in feature space. Detects semantic collapse.",
        "get_data": lambda df, report: get_position_separability(report),
        "plot": lambda data, df, report, cfg: _plot_position_separability(
            data,
            title=cfg["title"],
        ),
        "config": {"min_value": None},
    },

    "within_position_variance": {
        "title": "Within-position variance",
        "description": "Inconsistency of features within the same position group. Detects unstable role definitions.",
        "get_data": lambda df, report: get_within_position_variance(report),
        "plot": lambda data, df, report, cfg: _plot_within_position_variance(
            data,
            title=cfg["title"],
            top_n=cfg["top_n"],
        ),
        "config": {"top_n": 20, "min_value": None},
    },
}
