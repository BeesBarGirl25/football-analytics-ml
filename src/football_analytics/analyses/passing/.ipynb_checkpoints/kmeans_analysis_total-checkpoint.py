"""
Role taxonomy pipeline (RobustScaler -> PCA -> KMeans) with:
- K scan (inertia + silhouette)
- Stability scan (subsample ARI)
- Leakage tests (drop footedness / drop pct_other_pass)
- Label drift prevention: remap cluster IDs to a reference via centroid matching
- Auto role naming (rule-based) AFTER remap so names are stable
- Archetypal players per role (closest to centroid)
- Variant comparisons (ARI/AMI; permutation-safe)
- Archetype overlap (topN Jaccard)
- Find similar player (cosine in PCA space + explanation using role strengths + feature deltas)

Assumptions:
- parquet has: player_key, player, player_position, dataset (+ passing_feature_columns)
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from football_analytics.analyses.passing.features import passing_feature_columns

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics.pairwise import pairwise_distances  # ✅ FIXED IMPORT

# Optional (only if you want centroid-matching remap)
from scipy.optimize import linear_sum_assignment


# ----------------------------
# 0) Load + assemble datasets (Heroku-safe)
# ----------------------------
def load_player_level_datasets() -> pd.DataFrame:
    """
    Heroku: expected location is /app/artifacts/features, but we resolve from CWD.
    """
    base = Path.cwd() / "artifacts" / "features"
    files = sorted(base.glob("player_level_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No player_level_*.parquet files found under {base}")

    dfs: List[pd.DataFrame] = []
    for p in files:
        print(p)
        d = pd.read_parquet(p)
        d["dataset"] = p.stem.replace("player_level_", "")  # wc2018, euro2024, etc.
        dfs.append(d)

    return pd.concat(dfs, ignore_index=True)


# ----------------------------
# 0b) Collapse multiple tournaments -> ONE row per player_key
# ----------------------------
def _mode_or_first(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return None
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


def collapse_to_one_profile_per_player(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Produces one row per player_key across all tournaments.
    No minutes weighting: simple mean across tournaments (per90/pct already normalized).
    Drops goalkeepers here so they never enter modelling.
    """
    passing_cols = passing_feature_columns()

    keep_cols = [c for c in ["player_key", "player", "player_position", "dataset"] + passing_cols if c in df_raw.columns]
    df = df_raw[keep_cols].copy()

    # ✅ Drop keepers pre-modelling
    if "player_position" in df.columns:
        df = df[df["player_position"] != "Goalkeeper"].copy()

    # numeric feature columns
    num_cols = [c for c in passing_cols if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Backheels: NaN -> 0 (consistent with the rest of your pipeline)
    if "backheels_per90" in df.columns:
        df["backheels_per90"] = df["backheels_per90"].fillna(0.0)

    print("before groupby")
    for player in df["player"]:
        if "jude" in player.lower():
            print(player)

    df[num_cols] = df[num_cols].fillna(0.0)

    agg = {c: "mean" for c in num_cols}
    agg["player"] = _mode_or_first
    print("")
    print("agg[player]")
    print(agg['player'])
    agg["player_position"] = _mode_or_first

    out = df.groupby("player_key", as_index=False).agg(agg)
    print("")
    print("out")
    for player in out['player']:
        if "jude" in player.lower():
            print(player)
    # ✅ downstream should treat this as a single profile per player
    out["dataset"] = "ALL"
    return out


# ----------------------------
# 1) Feature selection + filter
# ----------------------------
FEATURES = [
    'passes_per_90','short_passes_per_90','medium_passes_per_90','long_passes_per_90',
    'crosses_per90','switches_per90','throughballs_per90','cutbacks_per90','backheels_per90',
    'passes_under_pressure_per90','key_passes_per90','assists_per90','pct_passes_under_pressure',
    'avg_pass_length','std_pass_length','pct_short_pass','pct_med_pass','pct_long_pass',
    'pct_ground_pass','pct_low_pass','pct_high_pass','pct_left_foot_pass','pct_right_foot_pass',
    'pct_head_pass','pct_foot_pass','pct_other_pass','pct_keeper_arm_pass',
    'pct_progressive_passes','pct_lateral_passes','pct_defensive_passes',
    'pct_pass_from_def_third','pct_pass_from_mid_third','pct_pass_from_att_third',
    'pct_pass_from_left_channel','pct_pass_from_central_channel','pct_pass_from_right_channel',
    'pct_pass_to_def_third','pct_pass_to_mid_third','pct_pass_to_left_channel',
    'pct_pass_to_central_channel','pct_pass_to_right_channel',
    'pct_passes_final_third','pct_passes_into_box',
    'pct_pass_from_zone_dl','pct_pass_to_zone_dl','pct_pass_from_zone_dc','pct_pass_to_zone_dc',
    'pct_pass_from_zone_dr','pct_pass_to_zone_dr','pct_pass_from_zone_ml','pct_pass_to_zone_ml',
    'pct_pass_from_zone_mc','pct_pass_to_zone_mc','pct_pass_from_zone_mr','pct_pass_to_zone_mr',
    'pct_pass_from_zone_al','pct_pass_to_zone_al','pct_pass_from_zone_ac','pct_pass_to_zone_ac',
    'pct_pass_from_zone_ar','pct_pass_to_zone_ar',
    'pct_pass_def_to_mid','pct_pass_def_to_att','pct_pass_mid_to_att','pct_pass_mid_to_mid',
    'pct_pass_att_to_mid','pct_pass_def_to_def','pct_pass_att_to_att',
    'pct_pass_left_to_centre','pct_pass_left_to_right','pct_pass_right_to_centre',
    'pct_pass_right_to_left','pct_pass_centre_to_left','pct_pass_centre_to_right',
    'pct_pass_wide_to_box','pct_pass_centre_to_box','pct_pass_def_to_box',
    'ttl_passes_F','pct_passes_F','passes_F_per90',
    'ttl_passes_FR','pct_passes_FR','passes_FR_per90',
    'ttl_passes_R','pct_passes_R','passes_R_per90',
    'ttl_passes_BR','pct_passes_BR','passes_BR_per90',
    'ttl_passes_B','pct_passes_B','passes_B_per90',
    'ttl_passes_BL','pct_passes_BL','passes_BL_per90',
    'ttl_passes_L','pct_passes_L','passes_L_per90',
    'ttl_passes_FL','pct_passes_FL','passes_FL_per90',
    'pass_angle_mean_overall','pass_angle_var_overall',
    'pass_angle_mean_def_third','pass_angle_var_def_third',
    'pass_angle_mean_mid_third','pass_angle_var_mid_third',
    'pass_angle_mean_att_third','pass_angle_var_att_third',
    'pass_angle_mean_left_channel','pass_angle_var_left_channel',
    'pass_angle_mean_centre_channel','pass_angle_var_centre_channel',
    'pass_angle_mean_right_channel','pass_angle_var_right_channel'
]

FOOTEDNESS_FEATURES = ["pct_left_foot_pass", "pct_right_foot_pass"]
ARTEFACTY_FEATURES = ["pct_other_pass"]


def build_model_df(df_raw: pd.DataFrame, features: list[str]):
    passing_cols = passing_feature_columns()

    # ✅ union of canonical + requested features, but only keep those present
    keep_cols = ["player_key", "player", "player_position", "dataset"] + list(dict.fromkeys(passing_cols + features))
    keep_cols = [c for c in keep_cols if c in df_raw.columns]

    df = df_raw[keep_cols].copy()

    # Filters
    df = df[df["player_position"].notna()].copy()
    if "has_minutes" in df.columns:
        df = df[df["has_minutes"] == 1]
    if "has_pass_events" in df.columns:
        df = df[df["has_pass_events"] == 1]
    if "passes_per_90" in df.columns:
        df = df[df["passes_per_90"].fillna(0) > 0]

    feats = [c for c in features if c in df.columns]

    if "backheels_per90" in feats:
        df["backheels_per90"] = df["backheels_per90"].fillna(0.0)

    df[feats] = df[feats].fillna(0.0)

    return df.reset_index(drop=True), feats


# ----------------------------------------
# 2) Scaling + PCA
# ----------------------------------------
def fit_pca(
    X_df: pd.DataFrame,
    n_components: int | None = None,
    var_threshold: float = 0.90,
    random_state: int = 42,
    make_plots: bool = False,
):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_df)

    if n_components is None:
        pca0 = PCA(n_components=min(50, X_scaled.shape[1]), random_state=random_state)
        pca0.fit(X_scaled)
        cum = np.cumsum(pca0.explained_variance_ratio_)
        n_components = int(np.argmax(cum >= var_threshold) + 1)

        if make_plots:
            plt.figure()
            plt.plot(cum)
            plt.axhline(var_threshold, linestyle="--")
            plt.xlabel("Number of PCs")
            plt.ylabel("Cumulative Explained Variance")
            plt.title(f"PCA cumulative variance (chosen n={n_components})")
            plt.show()

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    return scaler, pca, X_pca_df


# ----------------------------------------
# 3) K scans
# ----------------------------------------
def kmeans_scan(X_pca_df: pd.DataFrame, Ks: range, random_state: int = 42, n_init: int = 50):
    rows = []
    for k in Ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X_pca_df)
        sil = np.nan if np.unique(labels).size < 2 else silhouette_score(X_pca_df, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


# ----------------------------------------
# 4) Stability scan (subsample ARI)
# ----------------------------------------
def kmeans_stability(
    X_pca_df: pd.DataFrame,
    k: int,
    n_runs: int = 25,
    frac: float = 0.8,
    seed: int = 42,
    n_init: int = 20,
):
    rng = np.random.default_rng(seed)
    X_np = X_pca_df.to_numpy()
    n = X_np.shape[0]
    scores = []

    for i in range(n_runs):
        idx = rng.choice(n, int(frac * n), replace=False)
        X_sub = X_np[idx]
        l1 = KMeans(n_clusters=k, n_init=n_init, random_state=seed + i).fit_predict(X_sub)
        l2 = KMeans(n_clusters=k, n_init=n_init, random_state=seed + 10_000 + i).fit_predict(X_sub)
        scores.append(adjusted_rand_score(l1, l2))

    return float(np.mean(scores)), float(np.std(scores))


def stability_scan(X_pca_df: pd.DataFrame, Ks: range, **kwargs):
    rows = []
    for k in Ks:
        mean_ari, std_ari = kmeans_stability(X_pca_df, k, **kwargs)
        rows.append({"k": k, "stability_ari_mean": mean_ari, "stability_ari_std": std_ari})
    return pd.DataFrame(rows)


# ----------------------------------------
# 5) Fit roles + profiles
# ----------------------------------------
def fit_roles(
    df_model: pd.DataFrame,
    feats: list[str],
    X_pca_df: pd.DataFrame,
    k: int,
    random_state: int = 42,
    n_init: int = 50,
):
    km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X_pca_df)

    df_out = df_model.copy()
    df_out["role_id"] = labels

    std = StandardScaler()
    X_std = pd.DataFrame(std.fit_transform(df_out[feats]), columns=feats, index=df_out.index)

    df_out_std = pd.concat([df_out, X_std.add_prefix("z__")], axis=1)

    role_profiles = X_std.groupby(df_out["role_id"])[feats].mean().sort_index()

    return df_out_std, km, role_profiles, std


def print_role_profiles(role_profiles: pd.DataFrame, top_n: int = 10):
    for rid in role_profiles.index:
        s = role_profiles.loc[rid].sort_values(ascending=False)
        print(f"\nROLE {rid}")
        print("Top + traits")
        print(s.head(top_n))
        print("\nTop - traits")
        print(s.tail(top_n).sort_values())


# ----------------------------------------
# 6) Auto role naming (stable AFTER remap)
# ----------------------------------------
def _argmax_label(d: dict) -> tuple[str, float]:
    k = max(d, key=d.get)
    return k, float(d[k])


def role_family(s: pd.Series) -> str:
    if s.get("pct_pass_def_to_att", 0) > 2.0 and s.get("pct_long_pass", 0) > 1.0:
        return "Direct progressor / long-ball launcher"
    if s.get("throughballs_per90", 0) > 2.0 and s.get("key_passes_per90", 0) > 0.7:
        return "Needle-threading creator (throughball specialist)"
    if s.get("pct_pass_to_right_channel", 0) > 1.3 and s.get("pct_pass_from_right_channel", 0) > 1.3:
        return "Right-lane build-up fullback / outlet"
    if s.get("pct_pass_to_left_channel", 0) > 1.3 and s.get("pct_pass_from_left_channel", 0) > 1.3:
        return "Left-lane shaper / left-footed builder"
    if s.get("pct_pass_from_att_third", 0) > 0.9 and s.get("pct_passes_into_box", 0) > 0.7:
        return "Final-third facilitator (wide-to-box / combo)"
    if s.get("pct_pass_from_mid_third", 0) > 0.8 and s.get("pct_pass_mid_to_mid", 0) > 0.6:
        return "Mid-third circulator / switcher"
    if s.get("passes_per_90", 0) > 0.9 and s.get("short_passes_per_90", 0) > 0.7:
        return "High-volume metronome (possession hub)"
    if s.get("pct_other_pass", 0) > 2.5:
        return "Scrappy press-relief connector"
    if (
        s.get("pct_pass_from_def_third", 0) > 0.9
        and s.get("pct_pass_def_to_def", 0) > 0.9
        and s.get("pct_passes_final_third", 0) < -0.7
    ):
        return "Safe defensive circulator"
    if (
        s.get("pct_pass_from_att_third", 0) > 0.6
        and s.get("pct_passes_under_pressure", 0) > 0.8
        and s.get("pct_short_pass", 0) > 0.8
    ):
        return "Attacking link-up connector"
    return "Unlabelled"


def role_variant_suffix_human(s: pd.Series) -> str:
    from_lane, from_lane_val = _argmax_label({
        "left": s.get("pct_pass_from_left_channel", 0),
        "central": s.get("pct_pass_from_central_channel", 0),
        "right": s.get("pct_pass_from_right_channel", 0),
    })
    to_lane, to_lane_val = _argmax_label({
        "left": s.get("pct_pass_to_left_channel", 0),
        "central": s.get("pct_pass_to_central_channel", 0),
        "right": s.get("pct_pass_to_right_channel", 0),
    })
    third, third_val = _argmax_label({
        "deep": s.get("pct_pass_from_def_third", 0),
        "mid": s.get("pct_pass_from_mid_third", 0),
        "high": s.get("pct_pass_from_att_third", 0),
    })

    switchiness = s.get("pct_pass_left_to_right", 0) + s.get("pct_pass_right_to_left", 0)
    is_switcher = switchiness > 1.2
    is_recycler = s.get("pct_lateral_passes", 0) > 0.7

    is_threader = s.get("throughballs_per90", 0) > 1.0
    is_creator = s.get("key_passes_per90", 0) > 0.6
    is_box = (s.get("pct_passes_into_box", 0) > 0.7) or (s.get("pct_pass_wide_to_box", 0) > 0.7)

    is_progressive = s.get("pct_progressive_passes", 0) > 0.8
    is_long = (s.get("pct_long_pass", 0) > 0.9) or (s.get("avg_pass_length", 0) > 0.9)
    is_short = s.get("pct_short_pass", 0) > 0.9
    is_hub = s.get("passes_per_90", 0) > 1.0

    loc = []
    if third_val > 0.60:
        loc.append({"deep": "Deep", "mid": "Midfield", "high": "Attacking"}[third])
    if from_lane_val > 0.55:
        loc.append({"left": "Left", "central": "Central", "right": "Right"}[from_lane])
    if not loc and to_lane_val > 0.55:
        loc.append({"left": "Left", "central": "Central", "right": "Right"}[to_lane])

    parts = [" ".join(loc) if loc else "Mixed"]

    behaviour = []
    if is_threader and is_creator:
        behaviour.append("final-ball creator")
    elif is_threader:
        behaviour.append("through-ball specialist")
    elif is_creator:
        behaviour.append("chance creator")

    if is_box:
        behaviour.append("box-feeder")

    if is_hub:
        behaviour.append("tempo-setter")
    if is_switcher:
        behaviour.append("switcher")
    elif is_recycler:
        behaviour.append("recycler")

    if is_progressive and is_long:
        behaviour.append("direct progressor")
    elif is_progressive:
        behaviour.append("line-breaker")
    elif is_long:
        behaviour.append("long passer")
    elif is_short:
        behaviour.append("short connector")

    if behaviour:
        parts.append(" / ".join(behaviour[:2]))

    return " — ".join(parts)


def unique_role_name_map(role_profiles: pd.DataFrame) -> dict[int, str]:
    provisional = {}
    for rid in role_profiles.index:
        s = role_profiles.loc[rid]
        fam = role_family(s)
        if fam == "Unlabelled":
            fam = "Link-up connector"
        suffix = role_variant_suffix_human(s)
        provisional[rid] = f"{fam} — {suffix}"

    seen = {}
    out = {}
    for rid in sorted(provisional.keys()):
        nm = provisional[rid]
        if nm not in seen:
            seen[nm] = 1
            out[rid] = nm
        else:
            seen[nm] += 1
            out[rid] = f"{nm} #{seen[nm]}"
    return out


# ----------------------------------------
# 7) Prevent label drift: remap to reference centroids
# ----------------------------------------
def centroid_remap_feature_space(
    ref_df_roles: pd.DataFrame,
    new_df_roles: pd.DataFrame,
    feats_common: list[str],
) -> dict[int, int]:
    zcols = [f"z__{c}" for c in feats_common]

    ref_centroids = ref_df_roles.groupby("role_id")[zcols].mean().sort_index().to_numpy()
    new_centroids = new_df_roles.groupby("role_id")[zcols].mean().sort_index().to_numpy()

    cost = pairwise_distances(new_centroids, ref_centroids, metric="cosine")
    row_ind, col_ind = linear_sum_assignment(cost)

    ref_ids = np.array(sorted(ref_df_roles["role_id"].unique()))
    new_ids = np.array(sorted(new_df_roles["role_id"].unique()))

    return {int(new_ids[r]): int(ref_ids[c]) for r, c in zip(row_ind, col_ind)}


def apply_label_map(labels: np.ndarray, label_map: dict[int, int]) -> np.ndarray:
    return np.array([label_map[int(x)] for x in labels], dtype=int)


def remap_df_roles_to_reference_feature_space(
    df_roles: pd.DataFrame,
    feats: list[str],
    ref_df_roles: pd.DataFrame,
    ref_feats: list[str],
):
    feats_common = sorted(set(feats) & set(ref_feats))
    if not feats_common:
        raise ValueError("No common features between variant and reference.")

    zcols = [f"z__{c}" for c in feats_common]

    missing_ref = [c for c in zcols if c not in ref_df_roles.columns]
    missing_new = [c for c in zcols if c not in df_roles.columns]
    if missing_ref or missing_new:
        raise ValueError(
            f"Missing z__ cols. ref missing: {missing_ref[:5]} | new missing: {missing_new[:5]}"
        )

    label_map = centroid_remap_feature_space(
        ref_df_roles=ref_df_roles,
        new_df_roles=df_roles,
        feats_common=feats_common,
    )

    df_roles = df_roles.copy()
    df_roles["role_id"] = apply_label_map(df_roles["role_id"].to_numpy(), label_map)

    X_std_common = df_roles[zcols].copy()
    X_std_common.columns = feats_common
    role_profiles = X_std_common.groupby(df_roles["role_id"])[feats_common].mean().sort_index()

    return df_roles, role_profiles, label_map, feats_common


# ----------------------------------------
# 8) Role strengths + archetypes
# ----------------------------------------
def add_role_strengths(df_roles: pd.DataFrame, X_pca_df: pd.DataFrame, kmeans: KMeans):
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X_pca_df.columns)
    dist = pairwise_distances(X_pca_df, centroids, metric="cosine")
    strength = 1 / (1 + dist)

    strength_df = pd.DataFrame(strength, columns=[f"role_{i}_strength" for i in range(centroids.shape[0])])
    out = pd.concat([df_roles.reset_index(drop=True), strength_df], axis=1)
    out["role_strength"] = out.apply(lambda r: r[f"role_{int(r['role_id'])}_strength"], axis=1)
    return out


def top_archetypes(df_roles: pd.DataFrame, n: int = 10):
    for rid in sorted(df_roles["role_id"].unique()):
        sub = df_roles[df_roles["role_id"] == rid].sort_values("role_strength", ascending=False).head(n)
        print("\n", rid, sub["role_name"].iloc[0] if "role_name" in sub.columns else "")
        print(sub[["player", "player_position", "dataset", "role_strength"]])


# ----------------------------------------
# 9) Find similar player (fixed + safe)
# ----------------------------------------
_ROLE_COL_RE = re.compile(r"^role_(\d+)_strength$")


def build_similarity_index(df_roles: pd.DataFrame, X_pca_df: pd.DataFrame, feats: list[str], X_feat_df: pd.DataFrame):
    Xp = X_pca_df.to_numpy()
    sim = 1 - pairwise_distances(Xp, metric="cosine")

    # ✅ only role_{id}_strength, in numeric id order
    role_strength_cols = [c for c in df_roles.columns if _ROLE_COL_RE.match(c)]
    role_strength_cols = sorted(role_strength_cols, key=lambda c: int(_ROLE_COL_RE.match(c).group(1)))

    R = df_roles[role_strength_cols].to_numpy()
    F = X_feat_df[feats].to_numpy()

    name_to_idx = {p: i for i, p in enumerate(df_roles["player"].tolist())}

    return {
        "sim": sim,
        "role_strength_cols": role_strength_cols,
        "R": R,
        "F": F,
        "feats": feats,
        "name_to_idx": name_to_idx,
    }


def _top_role_ids(role_strength_cols: list[str], vec: np.ndarray, k: int) -> list[int]:
    order = np.argsort(vec)[::-1][:k]
    out: list[int] = []
    for j in order:
        m = _ROLE_COL_RE.match(role_strength_cols[j])
        if m:
            out.append(int(m.group(1)))
    return out


def find_similar_player(
    player_name: str,
    df_roles: pd.DataFrame,
    index_obj: dict,
    top_n: int = 10,
    role_topk: int = 3,
    diff_topk: int = 8,
):
    if player_name not in index_obj["name_to_idx"]:
        raise ValueError(f"Player not found: {player_name}")

    i = index_obj["name_to_idx"][player_name]
    sim = index_obj["sim"][i].copy()
    sim[i] = -np.inf

    order = np.argsort(sim)[::-1][:top_n]
    rows = []

    role_cols = index_obj["role_strength_cols"]
    R = index_obj["R"]
    F = index_obj["F"]
    feats = index_obj["feats"]

    i_top_roles = _top_role_ids(role_cols, R[i], role_topk)

    for j in order:
        j_top_roles = _top_role_ids(role_cols, R[j], role_topk)

        shared_ids = sorted(set(i_top_roles) & set(j_top_roles))
        shared_roles = []
        for rid in shared_ids:
            nm = df_roles[df_roles["role_id"] == rid]["role_name"].iloc[0] if "role_name" in df_roles.columns else str(rid)
            shared_roles.append(nm)

        deltas = np.abs(F[j] - F[i])
        top_idx = np.argsort(deltas)[::-1][:diff_topk]
        biggest = [f"{feats[t]} ({F[j][t] - F[i][t]:+.3f})" for t in top_idx]

        rows.append({
            "player": df_roles.loc[j, "player"],
            "dataset": df_roles.loc[j, "dataset"],
            "player_position": df_roles.loc[j, "player_position"],
            "similarity": float(sim[j]),
            "shared_roles": ", ".join(shared_roles) if shared_roles else "(none in top roles)",
            "biggest_differences": "; ".join(biggest),
        })

    return pd.DataFrame(rows)


# ----------------------------------------
# 10) Run variant
# ----------------------------------------
def run_variant(
    df_raw: pd.DataFrame,
    features: list[str],
    drop_features=None,
    k_final: int = 12,
    var_threshold: float = 0.90,
    random_state: int = 42,
    ref=None,
    make_plots: bool = False,
):
    drop_features = set(drop_features or [])
    df_model, feats = build_model_df(df_raw, features)
    feats = [c for c in feats if c not in drop_features]

    X_feat_df = df_model[feats]
    scaler, pca, X_pca_df = fit_pca(
        X_feat_df,
        n_components=None,
        var_threshold=var_threshold,
        random_state=random_state,
        make_plots=make_plots,
    )

    df_roles, km, role_profiles, std = fit_roles(
        df_model, feats, X_pca_df, k=k_final, random_state=random_state
    )

    label_map = None
    feats_common = None

    if ref is not None:
        df_roles, role_profiles, label_map, feats_common = remap_df_roles_to_reference_feature_space(
            df_roles=df_roles,
            feats=feats,
            ref_df_roles=ref["df_roles"],
            ref_feats=ref["feats"],
        )

    role_name_map = unique_role_name_map(role_profiles)
    df_roles["role_name"] = df_roles["role_id"].map(role_name_map)

    df_roles = add_role_strengths(df_roles, X_pca_df, km)

    return {
        "df_roles": df_roles,
        "role_profiles": role_profiles,
        "role_name_map": role_name_map,
        "label_map": label_map,
        "feats_common": feats_common,
        "scaler": scaler,
        "pca": pca,
        "kmeans": km,
        "std": std,
        "X_pca_df": X_pca_df,
        "X_feat_df": X_feat_df,
        "feats": feats,
    }


# =========================
# MAIN RUN (Heroku-safe)
# =========================
def main():
    df_raw = load_player_level_datasets()

    df_raw = collapse_to_one_profile_per_player(df_raw)

    Ks = range(2, 21)
    K_FINAL = 12

    base = run_variant(df_raw, FEATURES, drop_features=None, k_final=K_FINAL, var_threshold=0.90, ref=None, make_plots=False)
    nofoot = run_variant(df_raw, FEATURES, drop_features=FOOTEDNESS_FEATURES, k_final=K_FINAL, var_threshold=0.90, ref=base, make_plots=False)
    noother = run_variant(df_raw, FEATURES, drop_features=ARTEFACTY_FEATURES, k_final=K_FINAL, var_threshold=0.90, ref=base, make_plots=False)

    scan = kmeans_scan(base["X_pca_df"], Ks)
    print("Top silhouettes:")
    print(scan.sort_values("silhouette", ascending=False).head(10))

    stab = stability_scan(base["X_pca_df"], Ks, n_runs=25, frac=0.8, seed=42, n_init=20)
    print("Top stabilities:")
    print(stab.sort_values("stability_ari_mean", ascending=False).head(10))

    print("Role counts:")
    print(base["df_roles"]["role_id"].value_counts().sort_index())

    print_role_profiles(base["role_profiles"], top_n=10)
    top_archetypes(base["df_roles"], n=10)

    print("Variant comparison (proper, permutation-safe):")
    print(compare_clusterings(base["df_roles"], nofoot["df_roles"], "full vs nofoot"))
    print(compare_clusterings(base["df_roles"], noother["df_roles"], "full vs noother"))

    sim_index = build_similarity_index(
        df_roles=base["df_roles"],
        X_pca_df=base["X_pca_df"],
        feats=base["feats"],
        X_feat_df=base["X_feat_df"],
    )

    # result = find_similar_player("Cody Mathès Gakpo", base["df_roles"], sim_index, top_n=10)
    # print(result)


if __name__ == "__main__":
    main()
