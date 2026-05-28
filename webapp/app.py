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
    return render_template("index.html", active_page="discover")


@app.get("/player/<player_key>/<dataset>")
def player_page(player_key, dataset):
    return render_template("player.html", active_page="profile",
                           player_key=player_key, dataset=dataset)


@app.get("/compare")
def compare_page():
    return render_template(
        "compare.html",
        active_page="compare",
        src_key=request.args.get("src_key", ""),
        src_dataset=request.args.get("src_dataset", ""),
        dst_key=request.args.get("dst_key", ""),
        dst_dataset=request.args.get("dst_dataset", ""),
    )


@app.get("/leaderboard")
def leaderboard_page():
    return render_template("leaderboard.html", active_page="leaderboard")


@app.get("/competitions")
def competitions_page():
    return render_template("competitions.html", active_page="competitions")


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

                # --- Role stds for source player (used to normalise deltas → significance)
                cur.execute(
                    """
                    WITH role_players AS (
                        SELECT pf.features
                        FROM player_features pf
                        JOIN player p
                          ON p.model_version = pf.model_version
                         AND p.player_key    = pf.player_key
                         AND p.dataset       = pf.dataset
                        WHERE pf.model_version = %s AND p.role_id = %s
                    ),
                    role_stats AS (
                        SELECT i AS idx, STDDEV_POP((features)[i]) AS std
                        FROM role_players, generate_subscripts(features, 1) AS i
                        GROUP BY i
                    )
                    SELECT idx, std FROM role_stats ORDER BY idx
                    """,
                    (MODEL_VERSION, src_role_id),
                )
                _min_role_std = 1e-6
                _role_stds = {
                    int(idx) - 1: (float(std) if std is not None else 0.0)
                    for idx, std in cur.fetchall()
                }

                def _dz(i, delta):
                    std = _role_stds.get(i, 0.0)
                    if std < _min_role_std:
                        return None
                    return round(delta / std, 4)

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
                        {"feature": feat_names[i], "delta": float(d), "delta_z": _dz(i, d)}
                        for i, d in diff_sorted
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
                        {"feature": feat_names[i], "delta": float(d), "delta_z": _dz(i, d)}
                        for i, d in sim_sorted
                    ]

                    # fallback if everything got filtered out
                    if not greatest_similarities:
                        fallback = sorted(deltas, key=lambda x: abs(x[1]))[:min(sim_topk, len(deltas))]
                        greatest_similarities = [
                            {"feature": feat_names[i], "delta": float(d), "delta_z": _dz(i, d)}
                            for i, d in fallback
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

@app.get("/api/player_profile")
def player_profile():
    player_key = (request.args.get("player_key") or "").strip()
    dataset = (request.args.get("dataset") or "").strip()
    topk = int(request.args.get("topk", 12))
    min_std = float(request.args.get("min_std", 1e-6))  # avoid divide-by-zero weirdness

    if not player_key or not dataset:
        return jsonify({"error": "player_key and dataset are required"}), 400

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 1) metadata
            cur.execute(
                """
                SELECT player, player_position, role_id, role_name
                FROM player
                WHERE model_version=%s AND player_key=%s AND dataset=%s
                """,
                (MODEL_VERSION, player_key, dataset),
            )
            meta = cur.fetchone()
            if not meta:
                return jsonify({"error": "Player not found"}), 404

            player_name, pos, role_id, role_name = meta

            # 2) feats list (ordered)
            cur.execute("SELECT feats FROM model_version WHERE model_version=%s", (MODEL_VERSION,))
            row = cur.fetchone()
            if not row or row[0] is None:
                return jsonify({"error": "Model version feats not found"}), 500
            feat_names = row[0]  # list[str]

            # 3) player's feature vector
            cur.execute(
                """
                SELECT features
                FROM player_features
                WHERE model_version=%s AND player_key=%s AND dataset=%s
                """,
                (MODEL_VERSION, player_key, dataset),
            )
            frow = cur.fetchone()
            if not frow or frow[0] is None:
                return jsonify({"error": "Player features not found"}), 404

            feats = list(frow[0])
            if len(feats) != len(feat_names):
                return jsonify({"error": "Feature length mismatch"}), 500

            # 4) compute role stats + z-scores in Postgres
            #    - role_mean/std per feature index, based on all players in that role_id for this model_version
            #    - then z = (player_val - mean) / std
            #    NOTE: This assumes `player_features.features` is a Postgres array (float8[]).
            cur.execute(
                """
                WITH role_players AS (
                    SELECT pf.features
                    FROM player_features pf
                    JOIN player p
                      ON p.model_version = pf.model_version
                     AND p.player_key   = pf.player_key
                     AND p.dataset      = pf.dataset
                    WHERE pf.model_version = %s
                      AND p.role_id = %s
                ),
                role_stats AS (
                    SELECT
                        i AS idx,
                        AVG( (features)[i] ) AS mean,
                        STDDEV_POP( (features)[i] ) AS std
                    FROM role_players, generate_subscripts(features, 1) AS i
                    GROUP BY i
                )
                SELECT
                    s.idx,
                    s.mean,
                    s.std
                FROM role_stats s
                ORDER BY s.idx
                """,
                (MODEL_VERSION, role_id),
            )
            stats_rows = cur.fetchall()

            # if role has no stats (shouldn't happen), fallback to global
            if not stats_rows:
                cur.execute(
                    """
                    WITH all_players AS (
                        SELECT features
                        FROM player_features
                        WHERE model_version = %s
                    ),
                    stats AS (
                        SELECT
                            i AS idx,
                            AVG( (features)[i] ) AS mean,
                            STDDEV_POP( (features)[i] ) AS std
                        FROM all_players, generate_subscripts(features, 1) AS i
                        GROUP BY i
                    )
                    SELECT idx, mean, std
                    FROM stats
                    ORDER BY idx
                    """,
                    (MODEL_VERSION,),
                )
                stats_rows = cur.fetchall()

            # Build arrays aligned by idx (idx is 1-based in Postgres arrays)
            # We'll store in dict for safety.
            stats = {}
            for idx, mean, std in stats_rows:
                stats[int(idx)] = (float(mean) if mean is not None else 0.0, float(std) if std is not None else 0.0)

            # 5) score each feature by |z|, but only where std is valid
            cur.execute(
                "SELECT trait, category FROM trait_dictionary WHERE model_version = %s AND category IS NOT NULL",
                (MODEL_VERSION,),
            )
            trait_cats = {row[0]: row[1] for row in cur.fetchall()}

            scored = []
            cat_zs = {}
            for i0, (trait, val) in enumerate(zip(feat_names, feats)):
                idx = i0 + 1  # postgres index
                mean, std = stats.get(idx, (0.0, 0.0))
                if std is None or abs(std) < min_std:
                    continue
                z = (float(val) - float(mean)) / float(std)
                scored.append((trait, float(val), float(mean), float(std), float(z)))
                cat = trait_cats.get(trait)
                if cat:
                    cat_zs.setdefault(cat, []).append(z)

            category_summary = sorted(
                [
                    {"category": cat, "mean_z": round(sum(zs) / len(zs), 4), "feat_count": len(zs)}
                    for cat, zs in cat_zs.items()
                ],
                key=lambda x: -abs(x["mean_z"]),
            )

            scored.sort(key=lambda x: abs(x[4]), reverse=True)
            scored = scored[:max(1, topk)]

            # 6) enrich with dictionary in one query
            traits = [t for (t, *_rest) in scored]
            dict_map = {}
            if traits:
                cur.execute(
                    """
                    SELECT trait, display_name, description, category, higher_means
                    FROM trait_dictionary
                    WHERE model_version = %s
                      AND trait = ANY(%s)
                    """,
                    (MODEL_VERSION, traits),
                )
                for trait, display_name, description, category, higher_means in cur.fetchall():
                    dict_map[trait] = {
                        "display_name": display_name,
                        "description": description,
                        "category": category,
                        "higher_means": higher_means,
                    }

            enriched = []
            for trait, val, mean, std, z in scored:
                m = dict_map.get(trait, {})
                direction = "higher" if z >= 0 else "lower"
                enriched.append(
                    {
                        "trait": trait,
                        "display_name": m.get("display_name"),
                        "description": m.get("description"),
                        "category": m.get("category"),
                        "higher_means": m.get("higher_means"),
                        "value": val,
                        "role_mean": mean,
                        "role_std": std,
                        "z": z,
                        "direction_vs_role": direction,
                    }
                )

        return jsonify(
            {
                "player_key": player_key,
                "dataset": dataset,
                "player": player_name,
                "player_position": pos,
                "role_id": int(role_id),
                "role_name": role_name,
                "baseline": "role",
                "top_traits": enriched,
                "category_summary": category_summary,
            }
        )
    finally:
        conn.close()



if __name__ == "__main__":
    # local dev only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
