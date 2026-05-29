"""
Patches 'team' column into existing player-level parquets via StatsBomb lineups.
"""
from pathlib import Path
import pandas as pd
from statsbombpy import sb

COMPETITIONS = [
    (43,   3,   "player_level_wc2018.parquet"),
    (43,   106, "player_level_wc2022.parquet"),
    (55,   43,  "player_level_euro2020.parquet"),
    (55,   282, "player_level_euro2024.parquet"),
    (1267, 107, "player_level_afcon2023.parquet"),
    (223,  282, "player_level_copa2024.parquet"),
]

base = Path("artifacts/features")

for comp_id, season_id, filename in COMPETITIONS:
    path = base / filename
    if not path.exists():
        print(f"⚠️  {filename} not found, skipping")
        continue

    df = pd.read_parquet(path)
    print(f"\n--- {filename} ---")
    print(f"  rows: {len(df)}, player_key dtype: {df['player_key'].dtype}")
    print(f"  player_key samples: {df['player_key'].head(3).tolist()}")

    # Build player_id → team from lineups, keying everything as strings
    # to avoid int/float type mismatch
    player_team = {}
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    print(f"  matches: {len(matches)}")

    for match_id in matches["match_id"]:
        try:
            lineups = sb.lineups(match_id=match_id)
            for team_name, lineup_df in lineups.items():
                for pid in lineup_df["player_id"]:
                    player_team[str(pid)] = team_name
        except Exception as e:
            print(f"  ⚠️  match {match_id}: {e}")

    print(f"  lineup entries: {len(player_team)}, samples: {list(player_team.items())[:3]}")

    # Map using string keys on both sides
    df["team"] = df["player_key"].astype(str).map(player_team)

    # Fallback: strip trailing .0 from float-as-string keys (e.g. "3306.0" → "3306")
    if df["team"].isna().all():
        print("  direct map found nothing — trying int-cast fallback")
        def to_int_str(x):
            try:
                return str(int(float(x)))
            except Exception:
                return str(x)
        player_team_int = {to_int_str(k): v for k, v in player_team.items()}
        df["team"] = df["player_key"].apply(to_int_str).map(player_team_int)

    filled = df["team"].notna().sum()
    print(f"  matched: {filled}/{len(df)}")

    if filled > 0:
        df.to_parquet(path, index=False)
        print(f"  ✅ saved")
        print(f"  teams: {sorted(df['team'].dropna().unique())}")
    else:
        print(f"  ❌ no matches found — parquet NOT saved")
        print(f"  player_key as str samples: {df['player_key'].astype(str).head(3).tolist()}")
        print(f"  lineup key samples: {list(player_team.keys())[:3]}")
