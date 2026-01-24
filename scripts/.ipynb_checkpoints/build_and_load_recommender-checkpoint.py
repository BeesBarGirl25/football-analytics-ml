import os
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from football_analytics.analyses.passing.features import passing_feature_columns

# Import your pipeline functions
from football_analytics.notebooks.analyses.passing.kmeans_analysis_total import (
    load_player_level_datasets,
    run_variant,
    build_similarity_index,
)

MODEL_VERSION = os.getenv("MODEL_VERSION", "passing_v1")
TOP_N = int(os.getenv("TOP_N", "50"))
DATABASE_URL = os.environ["DATABASE_URL"]

# TODO: define this properly for your pipeline
# e.g. from football_analytics.analyses.passing.features import passing_feature_columns as FEATURES
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


def main():
    if FEATURES is None or isinstance(FEATURES, str):
        raise ValueError(
            "FEATURES is not defined as a list. "
            "Import your feature list and set FEATURES = [...]"
        )

    # 1) Run pipeline
    df_raw = load_player_level_datasets()

    base = run_variant(
        df_raw,
        FEATURES,
        drop_features=None,
        k_final=12,
        var_threshold=0.90,
        ref=None,
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
