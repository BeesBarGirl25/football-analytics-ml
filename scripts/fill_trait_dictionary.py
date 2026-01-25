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

CHANNEL_MAP = {
    "left": "left",
    "right": "right",
    "centre": "central",
    "central": "central",
}


# ---------------------------
# Regex patterns for trait families
# ---------------------------
RE_PASS_ANGLE_MEAN = re.compile(
    r"^pass_angle_mean_(overall|def_third|mid_third|att_third|left_channel|centre_channel|right_channel)$"
)
RE_PASS_ANGLE_VAR = re.compile(
    r"^pass_angle_var_(overall|def_third|mid_third|att_third|left_channel|centre_channel|right_channel)$"
)
RE_PASS_THIRD_TO_THIRD = re.compile(r"^pct_pass_(def|mid|att)_to_(def|mid|att)$")
RE_PASS_FROM_THIRD = re.compile(r"^pct_pass_from_(def|mid|att)_third$")
RE_PASS_TO_THIRD = re.compile(r"^pct_pass_to_(def|mid|att)_third$")
RE_PASS_FROM_CHANNEL = re.compile(r"^pct_pass_from_(left|right|central|centre)_channel$")
RE_PASS_TO_CHANNEL = re.compile(r"^pct_pass_to_(left|right|central|centre)_channel$")
RE_PASS_CHAN_TO_CHAN = re.compile(r"^pct_pass_(left|right|centre|central)_to_(left|right|centre|central)$")
RE_PCT_PASSES_DIR = re.compile(r"^pct_passes_([A-Z]{1,2})$")
RE_TTL_PASSES_DIR = re.compile(r"^ttl_passes_([A-Z]{1,2})$")
RE_PASSES_DIR_PER90 = re.compile(r"^passes_([A-Z]{1,2})_per90$")


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
    # 👇 more specific categories first
    if name.startswith("pass_angle_mean_") or name.startswith("pass_angle_var_"):
        return "Direction"
    if name.startswith("pct_pass_") and ("to" in tokens):
        return "Field progression"
    if name.startswith("pct_pass_from_") or name.startswith("pct_pass_to_"):
        return "Field progression"
    if name.startswith("pct_pass_") and ("left" in tokens or "right" in tokens or "centre" in tokens or "central" in tokens):
        return "Switch / width"
    if name in {"avg_pass_length", "std_pass_length"}:
        return "Pass length"
    if "throughballs" in tokens:
        return "Final ball"
    if "switches" in tokens:
        return "Switch / width"

    # 👇 existing logic
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
    if "high" in tokens or "medium" in tokens or "low" in tokens or "ground" in tokens:
        return "Pass height"
    if "long" in tokens or "short" in tokens:
        return "Pass length"
    if "switch" in tokens or "wide" in tokens:
        return "Switch / width"
    if "angle" in tokens or "direction" in tokens:
        return "Direction"
    return "General"


def _infer_display_name(trait: str) -> str:
    # Special-case nicer names for the families you surfaced
    m = RE_PASS_ANGLE_MEAN.match(trait)
    if m:
        scope = m.group(1).replace("_", " ")
        scope = scope.replace("centre", "central")
        if scope == "overall":
            return "Pass direction (mean) — overall"
        return f"Pass direction (mean) — {scope}"

    m = RE_PASS_ANGLE_VAR.match(trait)
    if m:
        scope = m.group(1).replace("_", " ")
        scope = scope.replace("centre", "central")
        if scope == "overall":
            return "Pass direction (variance) — overall"
        return f"Pass direction (variance) — {scope}"

    m = RE_PASS_THIRD_TO_THIRD.match(trait)
    if m:
        a, b = m.group(1), m.group(2)
        return _title(f"Passes: {a} third → {b} third (share)")

    m = RE_PASS_FROM_THIRD.match(trait)
    if m:
        a = m.group(1)
        return _title(f"Pass origin: {a} third (share)")

    m = RE_PASS_TO_THIRD.match(trait)
    if m:
        a = m.group(1)
        return _title(f"Pass destination: {a} third (share)")

    m = RE_PASS_FROM_CHANNEL.match(trait)
    if m:
        ch = CHANNEL_MAP.get(m.group(1), m.group(1))
        return _title(f"Pass origin: {ch} channel (share)")

    m = RE_PASS_TO_CHANNEL.match(trait)
    if m:
        ch = CHANNEL_MAP.get(m.group(1), m.group(1))
        return _title(f"Pass destination: {ch} channel (share)")

    m = RE_PASS_CHAN_TO_CHAN.match(trait)
    if m:
        a = CHANNEL_MAP.get(m.group(1), m.group(1))
        b = CHANNEL_MAP.get(m.group(2), m.group(2))
        return _title(f"Channel shift: {a} → {b} (share)")

    m = RE_PCT_PASSES_DIR.match(trait)
    if m:
        d = m.group(1)
        return _title(f"Pass direction: {DIR_MAP.get(d, d)} (share)")

    m = RE_TTL_PASSES_DIR.match(trait)
    if m:
        d = m.group(1)
        return _title(f"Pass direction: {DIR_MAP.get(d, d)} (total)")

    m = RE_PASSES_DIR_PER90.match(trait)
    if m:
        d = m.group(1)
        return _title(f"Pass direction: {DIR_MAP.get(d, d)} (per 90)")

    if trait == "pct_progressive_passes":
        return "Progressive passes (share)"
    if trait == "throughballs_per90":
        return "Through balls (per 90)"
    if trait == "switches_per90":
        return "Switches of play (per 90)"
    if trait == "avg_pass_length":
        return "Average pass length"
    if trait == "std_pass_length":
        return "Pass length variation"

    # Fallback: your original humaniser
    t = trait
    t = t.replace("pct_", "% ")
    t = t.replace("_per90", " per 90")
    t = t.replace("per_90", " per 90")

    tokens = _clean_tokens(t)

    rebuilt = []
    for tok in tokens:
        if tok in DIR_MAP:
            rebuilt.append(DIR_MAP[tok])
        elif tok in {"att", "mid", "def"} and "third" in tokens:
            rebuilt.append(THIRD_MAP.get(tok, tok))
        else:
            rebuilt.append(tok)

    s = " ".join(rebuilt)
    s = s.replace("% ", "% of ")
    s = s.replace(" into box", " into the box")
    s = s.replace(" final third", " into the final third")

    s = s.replace("pct pass", "% of passes")
    s = s.replace("pct passes", "% of passes")

    return _title(s)


def _infer_description(trait: str) -> tuple[str, str]:
    """
    returns (description, higher_means)
    """
    tokens = _clean_tokens(trait)

    # --- pass angle mean/variance
    m = RE_PASS_ANGLE_MEAN.match(trait)
    if m:
        scope = m.group(1)
        scope_txt = scope.replace("_", " ").replace("centre", "central")
        if scope_txt == "overall":
            scope_txt = "overall (all passes)"
        return (
            f"Average pass direction in degrees for {scope_txt}, using start→end pass vectors.",
            "Higher variance means less consistent direction; the mean reflects typical direction (sign depends on your angle convention).",
        )

    m = RE_PASS_ANGLE_VAR.match(trait)
    if m:
        scope = m.group(1).replace("_", " ").replace("centre", "central")
        if scope == "overall":
            scope = "overall (all passes)"
        return (
            f"Variation in pass direction for {scope} (circular variance).",
            "Higher means more varied pass directions; lower means a more consistent direction.",
        )

    # --- third-to-third transitions
    m = RE_PASS_THIRD_TO_THIRD.match(trait)
    if m:
        a, b = m.group(1), m.group(2)
        return (
            f"Share of passes that start in the {THIRD_MAP[a]} and end in the {THIRD_MAP[b]}.",
            "Higher means more of the player's passing follows this third-to-third pattern.",
        )

    # --- from/to thirds
    m = RE_PASS_FROM_THIRD.match(trait)
    if m:
        t = m.group(1)
        return (
            f"Share of passes that start in the {THIRD_MAP[t]}.",
            "Higher means more of their passing happens in that area of the pitch.",
        )

    m = RE_PASS_TO_THIRD.match(trait)
    if m:
        t = m.group(1)
        return (
            f"Share of passes that end in the {THIRD_MAP[t]}.",
            "Higher means they target that third more often with their passing.",
        )

    # --- from/to channels
    m = RE_PASS_FROM_CHANNEL.match(trait)
    if m:
        ch = CHANNEL_MAP.get(m.group(1), m.group(1))
        return (
            f"Share of passes that start in the {ch} channel (by pass origin).",
            "Higher means more of their passing originates from that channel.",
        )

    m = RE_PASS_TO_CHANNEL.match(trait)
    if m:
        ch = CHANNEL_MAP.get(m.group(1), m.group(1))
        return (
            f"Share of passes that end in the {ch} channel (by pass destination).",
            "Higher means they play into that channel more often.",
        )

    # --- channel-to-channel movement
    m = RE_PASS_CHAN_TO_CHAN.match(trait)
    if m:
        a = CHANNEL_MAP.get(m.group(1), m.group(1))
        b = CHANNEL_MAP.get(m.group(2), m.group(2))
        return (
            f"Share of passes that move the ball from the {a} channel to the {b} channel.",
            "Higher means they shift play between those channels more often.",
        )

    # --- direction buckets (pct + total + per90)
    m = RE_PCT_PASSES_DIR.match(trait)
    if m:
        d = m.group(1)
        d_txt = DIR_MAP.get(d, d)
        return (
            f"Share of passes in the {d_txt} direction bucket (based on pass angle).",
            "Higher means more passes played in that direction.",
        )

    m = RE_TTL_PASSES_DIR.match(trait)
    if m:
        d = m.group(1)
        d_txt = DIR_MAP.get(d, d)
        return (
            f"Total number of passes in the {d_txt} direction bucket.",
            "Higher means more volume of passes in that direction.",
        )

    m = RE_PASSES_DIR_PER90.match(trait)
    if m:
        d = m.group(1)
        d_txt = DIR_MAP.get(d, d)
        return (
            f"Number of passes per 90 minutes in the {d_txt} direction bucket.",
            "Higher means more passes in that direction per 90.",
        )

    # --- key actions / style traits
    if trait == "pct_progressive_passes":
        return (
            "Share of passes that meaningfully advance the ball toward goal (progression threshold).",
            "Higher means more progression with passing.",
        )

    if trait == "switches_per90":
        return (
            "Number of switches of play per 90 minutes (large lateral changes).",
            "Higher means more switching / changing the point of attack.",
        )

    if trait == "throughballs_per90":
        return (
            "Number of through balls per 90 minutes (passes into space behind the defence).",
            "Higher means more line-breaking final balls.",
        )

    if trait in {"avg_pass_length", "std_pass_length"}:
        if trait.startswith("avg"):
            return ("Average length of the player's passes.", "Higher means longer typical passes.")
        return ("Variation in pass length (standard deviation).", "Higher means a wider mix of pass lengths.")

    # ---------------------------
    # Fallbacks (your original logic, slightly safer wording)
    # ---------------------------

    # percent features
    if trait.startswith("pct_"):
        disp = _infer_display_name(trait)
        return (
            f"Share of the player's passes that match this trait ({disp}).",
            "The player does a greater share of their passing in this way.",
        )

    # per90 features
    if trait.endswith("_per90") or "per90" in tokens or ("per" in tokens and "90" in tokens):
        disp = _infer_display_name(trait)
        return (
            f"Rate per 90 minutes for this behaviour ({disp}).",
            "The player does this action more frequently per 90.",
        )

    # total / ttl features
    if trait.startswith("ttl_") or "total" in tokens:
        disp = _infer_display_name(trait)
        return (
            f"Total count for this behaviour ({disp}).",
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
    if "medium" in tokens or "med" in tokens:
        return ("How much the player uses medium-height passing.", "More medium-height passes.")
    if "low" in tokens:
        return ("How much the player uses low passing.", "More low passes.")
    if "ground" in tokens:
        return ("How much the player uses ground passes.", "More ground passes.")

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
