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
                SELECT player_key, dataset, player, role_name, player_position
                FROM player
                WHERE model_version = %s
                  AND player ILIKE %s
                ORDER BY player
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

        try:
            player_key = int(player_key)
        except (TypeError, ValueError):
            return jsonify({"error": "player_key must be an integer"}), 400

        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        n.dst_player_key,
                        n.dst_dataset,
                        n.similarity,
                        p.player,
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
                    ORDER BY n.rank
                    LIMIT %s
                    """,
                    (MODEL_VERSION, player_key, dataset, top_n),
                )
                neigh = cur.fetchall()

                if not neigh:
                    return jsonify({"results": []})

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
                src_feats = src_row[0]

                dst_pairs = [(int(r[0]), r[1]) for r in neigh]
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

                results = []
                for dst_key, dst_dataset, sim, name, role, pos in neigh:
                    dst_feats = dst_feat_map.get((int(dst_key), dst_dataset))
                    if dst_feats is None:
                        continue

                    deltas = [
                        (i, float(dst_feats[i]) - float(src_feats[i]))
                        for i in range(len(src_feats))
                    ]
                    deltas_sorted = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)[:diff_topk]

                    results.append(
                        {
                            "player_key": int(dst_key),
                            "dataset": dst_dataset,
                            "player": name,
                            "role": role,
                            "player_position": pos,
                            "similarity": float(sim),
                            "biggest_differences": [
                                {"feature_idx": i, "delta": d} for i, d in deltas_sorted
                            ],
                        }
                    )

                return jsonify({"results": results})
        finally:
            conn.close()

    except Exception as e:
        return jsonify(
            {"error": str(e), "trace": traceback.format_exc()},
        ), 500



if __name__ == "__main__":
    # local dev only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
