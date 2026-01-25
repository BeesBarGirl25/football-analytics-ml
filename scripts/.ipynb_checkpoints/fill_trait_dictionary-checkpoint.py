import os
import re
import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.environ["DATABASE_URL"]
MODEL_VERSION = os.getenv("MODEL_VERSION", "passing_v1")

# --- helpers

DIR_MAP = {
    "F": "forward",
    "B": "backward",
    "L": "left",
    "R": "right",
    "FL": "forward-left",
    "FR": "forward-right",
    "BL": "backward-left",
    "BR": "backward-right",
}

THIRD_MAP = {
    "def": "defensive third",
    "mid": "middle third",
    "att": "attacking third",
}

def _title(s: str) -> str:
    # nicer title-case but keep common acronyms
    words = s.split()
    out = []
    for w in words:
        if w.upper() in {"XG", "XA", "PCA"}:
            out.append(w.upper())
        else:
            out.append(w.capitalize())
    return " ".join(out)

def _clean_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[_\s]+", name.strip()) if t]

def _category(name: str, tokens: list[str]) -> str:
    if name.startswith("pct_"):
        return "Share / % breakdown"
    if name.endswith("_per90") or "per90" in tokens:
        return "Volume per 90"
    if name.startswith("ttl_") or "total" in tokens:
        return "Total volume"
    if "pressure" in tokens:
        return "Pressure"
    if "final" in tokens or "third" in tokens or "box" in tokens:
        return "Field progression"
    if "high" in tokens or "medium" in tokens or "low" in tokens:
        return "Pass height"
    if "long" in tokens or "short" in tokens:
        return "Pass length"
    if "switch" in tokens or "wide" in tokens:
        return "Switch / width"
    if "angle" in tokens or "direction" in tokens:
        return "Direction"
    return "General"

def _dir_phrase(token: str) -> str | None:
    return DIR_MAP.get(token)

def _infer_display_name(trait: str) -> str:
    t = trait

    # common patterns
    t = t.replace("pct_", "% ")
    t = t.replace("_per90", " per 90")
    t = t.replace("per_90", " per 90")

    tokens = _clean_tokens(t)

    # replace direction tokens
    rebuilt = []
    for tok in tokens:
        if tok in DIR_MAP:
            rebuilt.append(DIR_MAP[tok])
        elif tok in {"att", "mid", "def"} and "third" in tokens:
            rebuilt.append(THIRD_MAP.get(tok, tok))
        else:
            rebuilt.append(tok)

    # nice formatting
    s = " ".join(rebuilt)
    s = s.replace("% ", "% of ")
    s = s.replace(" passes ", " pass ")
    s = s.replace(" pass ", " passes ")
    s = s.replace(" into box", " into the box")
    s = s.replace(" final third", " into the final third")
    s = s.replace(" under pressure", " under pressure")
    s = s.replace(" to att", " to attacking third")
    s = s.replace(" from att", " from attacking third")
    s = s.replace(" to def", " to defensive third")
    s = s.replace(" from def", " from defensive third")
    s = s.replace(" to mid", " to middle third")
    s = s.replace(" from mid", " from middle third")

    # special cases you likely have
    s = s.replace("pct pass", "% of passes")
    s = s.replace("pct passes", "% of passes")

    return _title(s)

def _infer_description(trait: str) -> tuple[str, str]:
    """
    returns (description, higher_means)
    """
    tokens = _clean_tokens(trait)

    # percent features
    if trait.startswith("pct_"):
        base = trait[4:]
        disp = _infer_display_name(trait)
        return (
            f"Share of a player's passes that match this trait ({disp}).",
            "The player does a greater share of their passing in this way.",
        )

    # per90 features
    if trait.endswith("_per90") or "per90" in tokens or "per" in tokens and "90" in tokens:
        disp = _infer_display_name(trait)
        return (
            f"Rate per 90 minutes for this passing behaviour ({disp}).",
            "The player does this action more frequently per 90.",
        )

    # total / ttl features
    if trait.startswith("ttl_") or "total" in tokens:
        disp = _infer_display_name(trait)
        return (
            f"Total count for this passing behaviour ({disp}).",
            "The player has higher total volume in this behaviour.",
        )

    # pressure
    if "pressure" in tokens:
        return (
            "How much of the player's passing occurs under pressure.",
            "The player attempts a larger share of passes under pressure.",
        )

    # box / final third
    if "box" in tokens:
        return (
            "How much the player targets the penalty area with passing.",
            "More passes that enter or target the box.",
        )
    if "final" in tokens and "third" in tokens:
        return (
            "How much the player progresses play into the attacking third via passing.",
            "More passes into the final third.",
        )

    # height
    if "high" in tokens:
        return ("How much the player uses high/aerial passing.", "More high passes.")
    if "medium" in tokens:
        return ("How much the player uses medium-height passing.", "More medium-height passes.")
    if "low" in tokens:
        return ("How much the player uses low/ground passing.", "More low passes.")

    # length
    if "long" in tokens:
        return ("How much the player uses long passing.", "More long passes.")
    if "short" in tokens:
        return ("How much the player uses short passing.", "More short passes.")

    # default
    disp = _infer_display_name(trait)
    return (
        f"Player-level feature describing passing behaviour ({disp}).",
        "Higher values indicate more of this behaviour relative to peers.",
    )

def build_trait_meta(trait: str) -> dict:
    tokens = _clean_tokens(trait)
    display_name = _infer_display_name(trait)
    description, higher_means = _infer_description(trait)
    category = _category(trait, tokens)
    return {
        "trait": trait,
        "display_name": display_name,
        "description": description,
        "category": category,
        "higher_means": higher_means,
    }

def main():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT feats FROM model_version WHERE model_version = %s",
                    (MODEL_VERSION,),
                )
                row = cur.fetchone()
                if not row or row[0] is None:
                    raise RuntimeError(f"model_version.feats not found for {MODEL_VERSION}")

                feats = row[0]  # jsonb array -> list[str]
                metas = [build_trait_meta(f) for f in feats]

                rows = [
                    (
                        MODEL_VERSION,
                        m["trait"],
                        m["description"],
                        m["display_name"],
                        m["category"],
                        m["higher_means"],
                    )
                    for m in metas
                ]

                execute_values(
                    cur,
                    """
                    INSERT INTO trait_dictionary
                      (model_version, trait, description, display_name, category, higher_means)
                    VALUES %s
                    ON CONFLICT (model_version, trait) DO UPDATE
                    SET description  = EXCLUDED.description,
                        display_name = EXCLUDED.display_name,
                        category     = EXCLUDED.category,
                        higher_means = EXCLUDED.higher_means
                    """,
                    rows,
                )

        print(f"✅ Upserted {len(rows)} traits for model_version={MODEL_VERSION}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
