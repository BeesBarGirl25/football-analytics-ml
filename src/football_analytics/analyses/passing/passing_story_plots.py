from __future__ import annotations

from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from football_analytics.dqa.dqa_utils import DQAReport
from football_analytics.dqa.dqa_style import set_dqa_theme, apply_dqa_ax_style, dqa_accent

set_dqa_theme(font_scale=1.0)

PlotBundle = Dict[str, object]


# -----------------------------
# Helpers
# -----------------------------

def _get_extra(report: DQAReport, key: str) -> Any:
    if report.extras is None:
        return None
    return report.extras.get(key)


def _placeholder(title: str, text: str = "No data to plot") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout(pad=1.2)
    return fig


def _as_series(x: Any) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, dict):
        return pd.Series(x)
    try:
        return pd.Series(x)
    except Exception:
        return pd.Series(dtype=float)


def _as_frame(x: Any) -> pd.DataFrame:
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        return x.to_frame()
    try:
        return pd.DataFrame(x)
    except Exception:
        return pd.DataFrame()


# -----------------------------
# Story plots (matched to your outputs)
# -----------------------------

def plot_centroid_similarity_heatmap(
    sim: Any,
    *,
    title: str = "Position centroid similarity (cosine)",
) -> plt.Figure:
    """
    sim: DataFrame square matrix positions x positions from position_centroid_similarity().
    """
    mat = _as_frame(sim)
    if mat.empty:
        return _placeholder(title)

    # Size scales with number of positions
    n = max(mat.shape)
    size = max(6, min(14, 0.38 * n))
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
        mat,
        ax=ax,
        cmap="viridis",
        square=True,
        cbar=True,
        vmin=float(np.nanmin(mat.to_numpy())),
        vmax=float(np.nanmax(mat.to_numpy())),
    )

    ax.set_title(title)
    fig.tight_layout(pad=1.2)
    return fig


def plot_within_position_variance(
    v: Any,
    *,
    title: str = "Within-position variance (lower = more coherent)",
    top_n: int = 20,
) -> plt.Figure:
    """
    v: Series position -> variance score from within_position_variance().
    Your function sorts ascending already (lower variance = better coherence).
    """
    s = _as_series(v).dropna()
    if s.empty:
        return _placeholder(title)

    # For storytelling, show "most coherent" AND "least coherent" positions if lots exist.
    # Default behaviour: show worst (highest variance) unless you prefer the other direction.
    s = s.sort_values(ascending=False).head(top_n)

    fig_h = max(4, 0.35 * len(s) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(s.index.astype(str), s.values, color=dqa_accent())
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Variance score")

    apply_dqa_ax_style(ax)
    fig.tight_layout(pad=1.2)
    return fig


def plot_position_separability_score(
    score: Any,
    *,
    title: str = "Position separability (silhouette score)",
) -> plt.Figure:
    """
    score: float silhouette score from position_separability().
    Range is [-1, 1]. Rough heuristics:
      ~0     overlapping clusters
      >0.2   meaningful separation (often)
      <0     bad separation / points closer to other clusters
    """
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return _placeholder(title)

    try:
        val = float(score)
    except Exception:
        return _placeholder(title, "Separability score not numeric")

    fig, ax = plt.subplots(figsize=(10, 2.2))

    # Single horizontal bar from 0 baseline for readability.
    ax.barh(["silhouette"], [val], color=dqa_accent())

    ax.set_xlim(-1.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("Score (-1 to 1)")

    # Helpful guide lines for storytelling
    ax.axvline(0.0, linestyle="--", alpha=0.7)
    ax.axvline(0.2, linestyle="--", alpha=0.4)   # rough “meaningful” heuristic
    ax.axvline(0.4, linestyle="--", alpha=0.25)

    apply_dqa_ax_style(ax)
    fig.tight_layout(pad=1.2)
    return fig


def plot_feature_position_signal(
    signal: Any,
    *,
    title: str = "Feature position signal (what defines roles)",
    top_n: int = 20,
) -> plt.Figure:
    """
    signal: Series feature -> score from feature_position_signal().
    """
    s = _as_series(signal).dropna()
    if s.empty:
        return _placeholder(title)

    s = s.sort_values(ascending=False).head(top_n)

    fig_h = max(4, 0.35 * len(s) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(s.index.astype(str), s.values, color=dqa_accent())
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Effect-size-like score (higher = more positional signal)")

    apply_dqa_ax_style(ax)
    fig.tight_layout(pad=1.2)
    return fig


# -----------------------------
# Registry + rendering
# -----------------------------

STORY_PLOT_REGISTRY: Dict[str, Dict[str, object]] = {
    "position_centroid_similarity": {
        "title": "Position centroid similarity",
        "description": "Similarity between position archetypes (centroids) in your feature space. High similarity suggests overlapping roles (e.g., WB vs Winger).",
        "get_data": lambda df, report: _get_extra(report, "position_centroid_similarity"),
        "plot": lambda data, df, report, cfg: plot_centroid_similarity_heatmap(data, title=cfg["title"]),
        "config": {},
    },
    "position_separability": {
        "title": "Position separability",
        "description": "Single-number summary of how well positions form clusters in feature space (silhouette score, -1..1). Higher = more separable roles.",
        "get_data": lambda df, report: _get_extra(report, "position_separability"),
        "plot": lambda data, df, report, cfg: plot_position_separability_score(data, title=cfg["title"]),
        "config": {},
    },
    "within_position_variance": {
        "title": "Within-position variance",
        "description": "How internally consistent each position is. Higher variance suggests mixed role definitions or tactical freedom within that position group.",
        "get_data": lambda df, report: _get_extra(report, "within_position_variance"),
        "plot": lambda data, df, report, cfg: plot_within_position_variance(data, title=cfg["title"], top_n=cfg.get("top_n", 20)),
        "config": {"top_n": 20},
    },
    "feature_position_signal": {
        "title": "Feature position signal",
        "description": "Which features most strongly differentiate positions. Use this to narrate what defines fullbacks vs midfielders vs wingers in your model.",
        "get_data": lambda df, report: _get_extra(report, "feature_position_signal"),
        "plot": lambda data, df, report, cfg: plot_feature_position_signal(data, title=cfg["title"], top_n=cfg.get("top_n", 20)),
        "config": {"top_n": 20},
    },
}


def render_story_plot(
    key: str,
    df: pd.DataFrame,
    report: DQAReport,
) -> Optional[PlotBundle]:
    if key not in STORY_PLOT_REGISTRY:
        raise KeyError(f"Unknown story plot key: {key}")

    spec = STORY_PLOT_REGISTRY[key]
    cfg = dict(spec.get("config", {}))

    title = spec.get("title", key)
    description = spec.get("description", "")
    cfg["title"] = title

    data = spec["get_data"](df, report)
    if data is None:
        return None
    if isinstance(data, (pd.Series, pd.DataFrame)) and data.empty:
        return None

    fig = spec["plot"](data, df, report, cfg)
    plt.close(fig)

    return {"key": key, "title": title, "description": description, "fig": fig}


def render_story_dashboard(
    df: pd.DataFrame,
    report: DQAReport,
    keys: Optional[List[str]] = None,
) -> Dict[str, PlotBundle]:
    if keys is None:
        keys = [
            "position_separability",
            "position_centroid_similarity",
            "feature_position_signal",
            "within_position_variance",
        ]

    out: Dict[str, PlotBundle] = {}
    for k in keys:
        bundle = render_story_plot(k, df, report)
        if bundle is not None:
            out[k] = bundle
    return out
