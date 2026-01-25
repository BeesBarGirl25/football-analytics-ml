import os
import psycopg2
from flask import Flask, render_template, request, jsonify
from psycopg2.extras import execute_values
import traceback

DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_VERSION = os.getenv("MODEL_VERSION", "passing_v1")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

app = Flask(__name__)


def get_conn():
    return psycopg2.connect(DATABASE_URL)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/players")
def players():
    q = (request.args.get("q") or "")
    limit = int(request.args.get("limit", 10))

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (LOWER(player))
                  player_key, dataset, player, role_name, player_position
                FROM player
                WHERE model_version = %s
                  AND player ILIKE %s
                ORDER BY
                  LOWER(player),
                  CASE dataset
                    WHEN 'copa2024' THEN 6
                    WHEN 'euro2024' THEN 5
                    WHEN 'afcon2023' THEN 4
                    WHEN 'wc2022' THEN 3
                    WHEN 'euro2020' THEN 2
                    WHEN 'wc2018' THEN 1
                    ELSE 0
                  END DESC
                LIMIT %s
                """,
                (MODEL_VERSION, f"%{q}%", limit),
            )
            rows = cur.fetchall()

        return jsonify(
            [
                {
                    "player_key": r[0],
                    "dataset": r[1],
                    "player": r[2],
                    "role_name": r[3],
                    "player_position": r[4],
                }
                for r in rows
            ]
        )
    finally:
        conn.close()


@app.post("/api/recommend")
def recommend():
    try:
        payload = request.get_json(force=True) or {}

        player_key = payload.get("player_key")
        dataset = payload.get("dataset")
        top_n = int(payload.get("top_n", 10))
        diff_topk = int(payload.get("diff_topk", 8))

        if player_key is None or dataset is None:
            return jsonify({"error": "player_key and dataset are required"}), 400

        # IMPORTANT: keep as text because DB columns are text
        player_key = str(player_key).strip()
        dataset = str(dataset).strip()

        conn = get_conn()
        try:
            with conn.cursor() as cur:
                # --- Pull feature name list (ordered) from model_version (jsonb)
                cur.execute(
                    "SELECT feats FROM model_version WHERE model_version = %s",
                    (MODEL_VERSION,),
                )
                row = cur.fetchone()
                if not row or row[0] is None:
                    return jsonify({"error": "Model version feats not found"}), 500
                feat_names = row[0]  # list[str] (jsonb)

                # --- Source player metadata (for similarity section)
                cur.execute(
                    """
                    SELECT role_id, role_name, player, player_position
                    FROM player
                    WHERE model_version=%s AND player_key=%s AND dataset=%s
                    """,
                    (MODEL_VERSION, player_key, dataset),
                )
                src_meta = cur.fetchone()
                if not src_meta:
                    return jsonify({"error": "Source player not found"}), 404
                src_role_id, src_role_name, src_player_name, src_pos = src_meta

                # --- Neighbours
                # Key upgrades:
                # 1) Exclude exact self (key+dataset)
                # 2) Exclude same player name across other datasets (Harry Kane -> Harry Kane wc2022 etc)
                cur.execute(
                    """
                    SELECT
                        n.dst_player_key,
                        n.dst_dataset,
                        n.similarity,
                        p.player,
                        p.role_id,
                        p.role_name,
                        p.player_position
                    FROM player_neighbour n
                    JOIN player p
                      ON p.model_version = n.model_version
                     AND p.player_key = n.dst_player_key
                     AND p.dataset = n.dst_dataset
                    WHERE n.model_version = %s
                      AND n.src_player_key = %s
                      AND n.src_dataset = %s
                      AND NOT (n.dst_player_key = n.src_player_key AND n.dst_dataset = n.src_dataset)
                      AND LOWER(p.player) <> LOWER(%s)
                    ORDER BY n.rank
                    LIMIT %s
                    """,
                    (MODEL_VERSION, player_key, dataset, src_player_name, top_n),
                )

                neigh = cur.fetchall()

                # Return source even if no results
                if not neigh:
                    return jsonify(
                        {
                            "source": {
                                "player_key": player_key,
                                "dataset": dataset,
                                "player": src_player_name,
                                "player_position": src_pos,
                                "role_id": int(src_role_id),
                                "role_name": src_role_name,
                            },
                            "results": [],
                        }
                    )

                # --- Source features
                cur.execute(
                    """
                    SELECT features
                    FROM player_features
                    WHERE model_version = %s AND player_key = %s AND dataset = %s
                    """,
                    (MODEL_VERSION, player_key, dataset),
                )
                src_row = cur.fetchone()
                if not src_row:
                    return jsonify({"error": "Source player features not found"}), 404
                src_feats = src_row[0]  # list[float]

                # --- Neighbour features (bulk fetch)
                dst_pairs = [(r[0], r[1]) for r in neigh]
                dst_rows = [(MODEL_VERSION, k, d) for (k, d) in dst_pairs]

                execute_values(
                    cur,
                    """
                    SELECT pf.player_key, pf.dataset, pf.features
                    FROM player_features pf
                    JOIN (VALUES %s) AS v(model_version, player_key, dataset)
                      ON pf.model_version = v.model_version
                     AND pf.player_key = v.player_key
                     AND pf.dataset = v.dataset
                    """,
                    dst_rows,
                    template="(%s, %s, %s)",
                )

                dst_feat_rows = cur.fetchall()
                dst_feat_map = {(k, d): f for (k, d, f) in dst_feat_rows}

                # Safety: feature length alignment
                if len(src_feats) != len(feat_names):
                    return jsonify(
                        {
                            "error": f"Feature length mismatch: src_feats={len(src_feats)} feat_names={len(feat_names)}"
                        }
                    ), 500

                results = []

                # Add ranks to mimic notebook behaviour and make UI nicer
                for rank, (dst_key, dst_dataset, sim, name, dst_role_id, dst_role_name, pos) in enumerate(neigh, start=1):
                    dst_feats = dst_feat_map.get((dst_key, dst_dataset))
                    if dst_feats is None:
                        continue

                    # biggest differences by absolute delta
                    # deltas per feature
                    deltas = [
                        (i, float(dst_feats[i]) - float(src_feats[i]))
                        for i in range(len(src_feats))
                    ]
                    
                    # ---- biggest differences (largest abs delta)
                    diff_sorted = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)[:diff_topk]
                    biggest_differences = [{"feature": feat_names[i], "delta": d} for i, d in diff_sorted]
                    
                    # ---- greatest similarities (smallest abs delta, but not "both basically zero")
                    min_mag = float(payload.get("sim_min_mag", 0.05))   # allow override from UI if you want
                    sim_topk = int(payload.get("sim_topk", 8))          # allow override
                    
                    sim_candidates = []
                    for i, d in deltas:
                        src_v = float(src_feats[i])
                        dst_v = float(dst_feats[i])
                        if max(abs(src_v), abs(dst_v)) < min_mag:
                            continue  # too tiny to be meaningful
                        sim_candidates.append((i, d))
                    
                    sim_sorted = sorted(sim_candidates, key=lambda x: abs(x[1]))[:sim_topk]
                    greatest_similarities = [{"feature": feat_names[i], "delta": d} for i, d in sim_sorted]

                    # “why similar” chips (simple but effective)
                    why_similar = []
                    if int(dst_role_id) == int(src_role_id):
                        why_similar.append(f"Same role: {src_role_name}")

                    results.append(
                        {
                            "rank": rank,
                            "player_key": dst_key,
                            "dataset": dst_dataset,
                            "player": name,
                            "role_id": int(dst_role_id),
                            "role_name": dst_role_name,
                            "player_position": pos,
                            "similarity": float(sim),
                            "why_similar": why_similar,
                            "biggest_differences": biggest,
                            "greatest_similarities": greatest_similarities
                        }
                    )

                return jsonify(
                    {
                        "source": {
                            "player_key": player_key,
                            "dataset": dataset,
                            "player": src_player_name,
                            "player_position": src_pos,
                            "role_id": int(src_role_id),
                            "role_name": src_role_name,
                        },
                        "results": results,
                    }
                )

        finally:
            conn.close()

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    # local dev only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
