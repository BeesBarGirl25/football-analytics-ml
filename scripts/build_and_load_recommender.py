import os
import numpy as np
import psycopg2
from psycopg2.extras import execute_values, Json

from football_analytics.analyses.passing.kmeans_analysis_total import (
    load_player_level_datasets,
    collapse_to_one_profile_per_player,  
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

def _find_player(df, needle="bellingham"):
    m = df["player"].astype(str).str.contains(needle, case=False, na=False)
    return df.loc[m].copy()

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

    j = _find_player(df_roles, "bellingham")
    print("Jude in df_roles:", len(j))
    print(j[["player_key","dataset","player","player_position","role_id","role_name"]].head(10))
    
    assert len(j) == 1, f"Expected 1 Jude row in df_roles, got {len(j)}"
    j_idx = int(j.index[0])
    
    print("Jude PCA row exists:", j_idx < len(X_pca_df), "shape", X_pca_df.shape)
    print("Jude FEAT row exists:", j_idx < len(X_feat_df), "shape", X_feat_df.shape)

    
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
                    if isinstance(r.get("player"), str) and "bellingham" in r["player"].lower():
                        print("ABOUT TO INSERT JUDE:")
                        print(" raw key:", r.get("player_key"), type(r.get("player_key")))
                        print(" _as_int:", _as_int(r.get("player_key")))
                        print(" dataset:", r.get("dataset"))
                        print(" pos:", r.get("player_position"))

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
                            str(r["team"]) if "team" in r and r["team"] is not None else "",
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
                j_keys = [row for row in player_rows if "bellingham" in row[3].lower()]
                print("Jude rows in player_rows:", len(j_keys))
                print(j_keys[:1])
                assert len(j_keys) == 1, "Jude not in player_rows — lost during row construction"

                # 4a) player
                execute_values(
                    cur,
                    """
                    INSERT INTO player (
                        model_version, player_key, dataset, player, player_position, role_id, role_name, team
                    ) VALUES %s
                    ON CONFLICT (model_version, player_key, dataset) DO UPDATE
                    SET player = EXCLUDED.player,
                        player_position = EXCLUDED.player_position,
                        role_id = EXCLUDED.role_id,
                        role_name = EXCLUDED.role_name,
                        team = EXCLUDED.team
                    """,
                    player_rows,
                )
                cur.execute(
                    """
                    SELECT player_key, dataset, player
                    FROM player
                    WHERE model_version=%s AND LOWER(player) LIKE '%%bellingham%%'
                    """,
                    (MODEL_VERSION,),
                )
                print("DB Jude after player insert:", cur.fetchall())


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
                cur.execute(
                    """
                    SELECT pf.player_key, pf.dataset, array_length(pf.features, 1)
                    FROM player_features pf
                    JOIN player p
                      ON p.model_version=pf.model_version AND p.player_key=pf.player_key AND p.dataset=pf.dataset
                    WHERE pf.model_version=%s AND LOWER(p.player) LIKE '%%bellingham%%'
                    """,
                    (MODEL_VERSION,),
                )
                print("DB Jude after features insert:", cur.fetchall())


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
