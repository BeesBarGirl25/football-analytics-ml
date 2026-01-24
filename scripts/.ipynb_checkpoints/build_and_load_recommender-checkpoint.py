import os
import numpy as np
import psycopg2
from psycopg2.extras import execute_values, Json

from football_analytics.analyses.passing.kmeans_analysis_total import (
    load_player_level_datasets,
    collapse_to_one_profile_per_player,  # ✅ NEW: aggregate across tournaments + drop GKs
    run_variant,
    build_similarity_index,
    FEATURES,
)

MODEL_VERSION = os.getenv("MODEL_VERSION", "passing_v1")
TOP_N = int(os.getenv("TOP_N", "50"))
DATABASE_URL = os.environ["DATABASE_URL"]


def _as_int(x):
    # Handles floats like 2941.0 safely
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    # if it comes as a string, try int conversion
    return int(str(x))


def _as_float_list(row) -> list[float]:
    # Ensures pure Python floats (not numpy types)
    return [float(v) for v in row]


def main():
    # 1) Load raw datasets + collapse to ONE profile per player (across tournaments)
    #    Also drops Goalkeepers inside the collapse function (per your requirement).
    df_raw = load_player_level_datasets()
    df_raw = collapse_to_one_profile_per_player(df_raw)  # ✅ NEW BEHAVIOUR

    # 2) Run pipeline (NO PLOTS)
    base = run_variant(
        df_raw,
        FEATURES,
        drop_features=None,
        k_final=12,
        var_threshold=0.90,
        ref=None,
        make_plots=False,
    )

    df_roles = base["df_roles"].reset_index(drop=True)
    X_pca_df = base["X_pca_df"].reset_index(drop=True)
    X_feat_df = base["X_feat_df"].reset_index(drop=True)
    feats = base["feats"]

    sim_index = build_similarity_index(
        df_roles=df_roles,
        X_pca_df=X_pca_df,
        feats=feats,
        X_feat_df=X_feat_df,
    )
    sim = sim_index["sim"]

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                # 3) Upsert model_version
                # feats column is jsonb ✅
                cur.execute(
                    """
                    INSERT INTO model_version (model_version, notes, k_final, pca_components, feats)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (model_version) DO UPDATE
                    SET notes = EXCLUDED.notes,
                        k_final = EXCLUDED.k_final,
                        pca_components = EXCLUDED.pca_components,
                        feats = EXCLUDED.feats
                    """,
                    (
                        MODEL_VERSION,
                        "Passing recommender",
                        12,
                        int(X_pca_df.shape[1]),
                        Json(feats),  # ✅ jsonb
                    ),
                )

                # 4) Bulk rows
                player_rows = []
                emb_rows = []
                feat_rows = []

                for i, r in df_roles.iterrows():
                    key = _as_int(r["player_key"])
                    dataset = str(r["dataset"])  # will be "ALL" after collapse

                    player_rows.append(
                        (
                            MODEL_VERSION,
                            key,
                            dataset,
                            str(r["player"]),
                            str(r["player_position"]),
                            int(r["role_id"]),
                            str(r["role_name"]),
                        )
                    )

                    # embedding/features columns are float8[] ✅
                    emb_rows.append(
                        (
                            MODEL_VERSION,
                            key,
                            dataset,
                            _as_float_list(X_pca_df.iloc[i].to_numpy()),
                        )
                    )

                    feat_rows.append(
                        (
                            MODEL_VERSION,
                            key,
                            dataset,
                            _as_float_list(X_feat_df.iloc[i].to_numpy()),
                        )
                    )

                # 4a) player
                execute_values(
                    cur,
                    """
                    INSERT INTO player (
                        model_version, player_key, dataset, player, player_position, role_id, role_name
                    ) VALUES %s
                    ON CONFLICT (model_version, player_key, dataset) DO UPDATE
                    SET player = EXCLUDED.player,
                        player_position = EXCLUDED.player_position,
                        role_id = EXCLUDED.role_id,
                        role_name = EXCLUDED.role_name
                    """,
                    player_rows,
                )

                # 4b) player_embedding (float8[])
                execute_values(
                    cur,
                    """
                    INSERT INTO player_embedding (model_version, player_key, dataset, embedding)
                    VALUES %s
                    ON CONFLICT (model_version, player_key, dataset) DO UPDATE
                    SET embedding = EXCLUDED.embedding
                    """,
                    emb_rows,
                )

                # 4c) player_features (float8[])
                execute_values(
                    cur,
                    """
                    INSERT INTO player_features (model_version, player_key, dataset, features)
                    VALUES %s
                    ON CONFLICT (model_version, player_key, dataset) DO UPDATE
                    SET features = EXCLUDED.features
                    """,
                    feat_rows,
                )

                # 5) neighbours
                neigh_rows = []
                for i, r in df_roles.iterrows():
                    sims = sim[i].copy()
                    sims[i] = -np.inf
                    order = np.argsort(sims)[::-1][:TOP_N]

                    src_key = _as_int(r["player_key"])
                    src_dataset = str(r["dataset"])  # will be "ALL"

                    for rank, j in enumerate(order, start=1):
                        neigh_rows.append(
                            (
                                MODEL_VERSION,
                                src_key,
                                src_dataset,
                                _as_int(df_roles.loc[j, "player_key"]),
                                str(df_roles.loc[j, "dataset"]),
                                int(rank),
                                float(sims[j]),
                            )
                        )

                execute_values(
                    cur,
                    """
                    INSERT INTO player_neighbour (
                        model_version, src_player_key, src_dataset,
                        dst_player_key, dst_dataset, rank, similarity
                    )
                    VALUES %s
                    ON CONFLICT (model_version, src_player_key, src_dataset, rank) DO UPDATE
                    SET dst_player_key = EXCLUDED.dst_player_key,
                        dst_dataset = EXCLUDED.dst_dataset,
                        similarity = EXCLUDED.similarity
                    """,
                    neigh_rows,
                )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
