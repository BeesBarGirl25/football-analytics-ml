import os
import traceback

import psycopg2
from flask import Flask, render_template, request, jsonify

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


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


@app.post("/api/recommend")
def recommend():
    try:
        payload = request.get_json(force=True) or {}

        player_key = payload.get("player_key")
        dataset = payload.get("dataset")
        top_n = int(payload.get("top_n", 10))
        diff_topk = int(payload.get("diff_topk", 8))

        # optional knobs for "greatest similarities"
        sim_topk = int(payload.get("sim_topk", diff_topk))
        sim_min_mag = float(payload.get("sim_min_mag", 0.05))  # ignore near-zero vs near-zero

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

                # --- Source player metadata
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

                # --- Neighbours (exclude self; exclude same name across datasets)
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
                if not src_row or src_row[0] is None:
                    return jsonify({"error": "Source player features not found"}), 404

                src_feats = list(src_row[0])  # list[float]

                if len(src_feats) != len(feat_names):
                    return jsonify(
                        {
                            "error": (
                                f"Feature length mismatch: "
                                f"src_feats={len(src_feats)} feat_names={len(feat_names)}"
                            )
                        }
                    ), 500

                # --- Neighbour features (bulk fetch via VALUES)
                dst_pairs = [(str(r[0]), str(r[1])) for r in neigh]  # (player_key, dataset)

                # Build VALUES list safely
                values_sql = b",".join(
                    cur.mogrify("(%s,%s,%s)", (MODEL_VERSION, k, d)) for (k, d) in dst_pairs
                )
                cur.execute(
                    b"""
                    SELECT pf.player_key, pf.dataset, pf.features
                    FROM player_features pf
                    JOIN (VALUES
                    """ + values_sql + b"""
                    ) AS v(model_version, player_key, dataset)
                      ON pf.model_version = v.model_version
                     AND pf.player_key = v.player_key
                     AND pf.dataset = v.dataset
                    """,
                )
                dst_feat_rows = cur.fetchall()
                dst_feat_map = {(str(k), str(d)): f for (k, d, f) in dst_feat_rows}

                results = []

                for rank, (dst_key, dst_dataset, sim, name, dst_role_id, dst_role_name, pos) in enumerate(
                    neigh, start=1
                ):
                    dst_key = str(dst_key)
                    dst_dataset = str(dst_dataset)

                    dst_feats = dst_feat_map.get((dst_key, dst_dataset))
                    if dst_feats is None:
                        continue
                    dst_feats = list(dst_feats)

                    # deltas per feature
                    deltas = [
                        (i, _safe_float(dst_feats[i]) - _safe_float(src_feats[i]))
                        for i in range(len(src_feats))
                    ]

                    # ---- biggest differences (largest abs delta)
                    diff_sorted = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)[:diff_topk]
                    biggest_differences = [
                        {"feature": feat_names[i], "delta": float(d)} for i, d in diff_sorted
                    ]

                    # ---- greatest similarities (smallest abs delta, avoid "both ~0")
                    sim_candidates = []
                    for i, d in deltas:
                        src_v = _safe_float(src_feats[i])
                        dst_v = _safe_float(dst_feats[i])
                        if max(abs(src_v), abs(dst_v)) < sim_min_mag:
                            continue
                        sim_candidates.append((i, d))

                    sim_sorted = sorted(sim_candidates, key=lambda x: abs(x[1]))[:sim_topk]
                    greatest_similarities = [
                        {"feature": feat_names[i], "delta": float(d)} for i, d in sim_sorted
                    ]

                    # fallback if everything got filtered out
                    if not greatest_similarities:
                        fallback = sorted(deltas, key=lambda x: abs(x[1]))[:min(sim_topk, len(deltas))]
                        greatest_similarities = [
                            {"feature": feat_names[i], "delta": float(d)} for i, d in fallback
                        ]

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
                            "biggest_differences": biggest_differences,
                            "greatest_similarities": greatest_similarities,
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


@app.get("/api/trait")
def trait_one():
    trait = (request.args.get("trait") or "").strip()
    if not trait:
        return jsonify({"error": "trait is required"}), 400

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trait, display_name, description, category, higher_means
                FROM trait_dictionary
                WHERE model_version = %s AND trait = %s
                """,
                (MODEL_VERSION, trait),
            )
            row = cur.fetchone()

        if not row:
            return jsonify(
                {
                    "trait": trait,
                    "display_name": None,
                    "description": "No glossary entry yet.",
                    "category": None,
                    "higher_means": None,
                }
            )

        return jsonify(
            {
                "trait": row[0],
                "display_name": row[1],
                "description": row[2],
                "category": row[3],
                "higher_means": row[4],
            }
        )
    finally:
        conn.close()


@app.post("/api/traits")
def traits_bulk():
    payload = request.get_json(force=True) or {}
    traits = payload.get("traits") or []

    if not isinstance(traits, list) or not traits:
        return jsonify({"traits": {}})

    # de-dupe + stringify + cap
    traits = sorted({str(t).strip() for t in traits if t is not None and str(t).strip()})
    traits = traits[:500]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trait, display_name, description, category, higher_means
                FROM trait_dictionary
                WHERE model_version = %s
                  AND trait = ANY(%s)
                """,
                (MODEL_VERSION, traits),
            )
            rows = cur.fetchall()

        out = {}
        for trait, display_name, description, category, higher_means in rows:
            out[trait] = {
                "trait": trait,
                "display_name": display_name,
                "description": description,
                "category": category,
                "higher_means": higher_means,
            }

        # fill missing with safe defaults
        for t in traits:
            out.setdefault(
                t,
                {
                    "trait": t,
                    "display_name": None,
                    "description": "No glossary entry yet.",
                    "category": None,
                    "higher_means": None,
                },
            )

        return jsonify({"traits": out})
    finally:
        conn.close()


if __name__ == "__main__":
    # local dev only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
