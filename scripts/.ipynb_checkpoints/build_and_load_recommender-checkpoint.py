import os
import numpy as np
import psycopg2
from psycopg2.extras import execute_values, Json

# Import your pipeline functions + FEATURES list (single source of truth)
from football_analytics.analyses.passing.kmeans_analysis_total import (
    load_player_level_datasets,
    run_variant,
    build_similarity_index,
    FEATURES,
)

MODEL_VERSION = os.getenv("MODEL_VERSION", "passing_v1")
TOP_N = int(os.getenv("TOP_N", "50"))
DATABASE_URL = os.environ["DATABASE_URL"]


def main():
    # 1) Run pipeline (NO PLOTS)
    df_raw = load_player_level_datasets()

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

    # 2) DB writes
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                # 2a) Upsert model_version (feats is jsonb)
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

                # 3) Build rows for bulk upserts
                player_rows = []
                emb_rows = []
                feat_rows = []

                for i, r in df_roles.iterrows():
                    key = r["player_key"]
                    dataset = r["dataset"]

                    player_rows.append(
                        (
                            MODEL_VERSION,
                            key,
                            dataset,
                            r["player"],
                            r["player_position"],
                            int(r["role_id"]),
                            r["role_name"],
                        )
                    )

                    # ✅ If embedding/features columns are jsonb, wrap with Json(...)
                    emb_rows.append(
                        (
                            MODEL_VERSION,
                            key,
                            dataset,
                            Json(list(X_pca_df.iloc[i].to_numpy())),
                        )
                    )

                    feat_rows.append(
                        (
                            MODEL_VERSION,
                            key,
                            dataset,
                            Json(list(X_feat_df.iloc[i].to_numpy())),
                        )
                    )

                # 3a) Upsert players
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

                # 3b) Upsert embeddings (jsonb)
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

                # 3c) Upsert features (jsonb)
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

                # 4) Upsert neighbours
                neigh_rows = []
                for i, r in df_roles.iterrows():
                    sims = sim[i].copy()
                    sims[i] = -np.inf
                    order = np.argsort(sims)[::-1][:TOP_N]

                    for rank, j in enumerate(order, start=1):
                        neigh_rows.append(
                            (
                                MODEL_VERSION,
                                r["player_key"],
                                r["dataset"],
                                df_roles.loc[j, "player_key"],
                                df_roles.loc[j, "dataset"],
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
