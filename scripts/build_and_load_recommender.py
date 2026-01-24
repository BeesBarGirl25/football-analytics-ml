import os
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from football_analytics.analyses.passing.features import passing_feature_columns

# Import your pipeline functions
from football_analytics.analyses.passing.kmeans_analysis_total import (
    load_player_level_datasets,
    run_variant,
    build_similarity_index,
    FEATURES,  # ✅ import the list instead of redefining it
)

MODEL_VERSION = os.getenv("MODEL_VERSION", "passing_v1")
TOP_N = int(os.getenv("TOP_N", "50"))
DATABASE_URL = os.environ["DATABASE_URL"]

def main():
    if FEATURES is None or isinstance(FEATURES, str):
        raise ValueError(
            "FEATURES is not defined as a list. "
            "Import your feature list and set FEATURES = [...]"
        )

    # 1) Run pipeline
    df_raw = load_player_level_datasets()

    base = run_variant(df_raw, FEATURES, drop_features=None, k_final=12, var_threshold=0.90, ref=None, make_plots=False)

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
                # 2a) Insert model_version
                cur.execute(
                    """
                    INSERT INTO model_version (model_version, notes, k_final, pca_components, feats)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (model_version) DO NOTHING
                    """,
                    (MODEL_VERSION, "Passing recommender", 12, X_pca_df.shape[1], feats),
                )

                # 3) Build rows for bulk insert
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

                    emb_rows.append(
                        (
                            MODEL_VERSION,
                            key,
                            dataset,
                            list(X_pca_df.iloc[i].to_numpy()),
                        )
                    )

                    feat_rows.append(
                        (
                            MODEL_VERSION,
                            key,
                            dataset,
                            list(X_feat_df.iloc[i].to_numpy()),
                        )
                    )

                # 3a) Insert players
                execute_values(
                    cur,
                    """
                    INSERT INTO player (
                        model_version, player_key, dataset, player, player_position, role_id, role_name
                    ) VALUES %s
                    ON CONFLICT (model_version, player_key, dataset) DO UPDATE
                    SET role_id = EXCLUDED.role_id,
                        role_name = EXCLUDED.role_name
                    """,
                    player_rows,
                )

                # 3b) Insert embeddings
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

                # 3c) Insert features
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

                # 4) Insert neighbours
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
                                rank,
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

        # conn commits automatically because of "with conn:"
    finally:
        conn.close()


if __name__ == "__main__":
    main()
