from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  
    sns = None

# -----------------------------
# Theming
# -----------------------------

DQA_COLORS = {
    "bg": "#0f1218",        
    "panel": "#121826",     
    "grid": "#2a3242",      
    "text": "#e8edf5",      
    "muted": "#9aa7bd",     
    "accent": "#3b82f6",    
    "warn": "#f59e0b",      
    "bad": "#ef4444",       
}

def set_dqa_theme(*, font_scale: float = 1.0) -> None:
    
    if sns is not None:
        sns.set_theme(style="darkgrid", context="notebook", font_scale=font_scale)

    c = DQA_COLORS

    mpl.rcParams.update({
        "figure.facecolor": c["bg"],
        "axes.facecolor": c["panel"],
        "savefig.facecolor": c["bg"],

        "text.color": c["text"],
        "axes.labelcolor": c["text"],
        "axes.titlecolor": c["text"],
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],

        "grid.color": c["grid"],
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,

        "axes.edgecolor": c["grid"],
        "axes.spines.top": False,
        "axes.spines.right": False,

        "legend.frameon": False,

        "figure.dpi": 120,
        "savefig.dpi": 200,
        "figure.autolayout": False,
    })


def apply_dqa_ax_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="x", alpha=0.6)
    ax.grid(False, axis="y")
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)


def dqa_accent() -> str:
    return DQA_COLORS["accent"]