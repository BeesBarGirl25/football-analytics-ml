"""
Adds a 'team' column to existing player-level parquets using StatsBomb lineup
data. Much faster than rebuilding from events — lineups are small files.
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
    print(f"\n{filename}: {len(df)} players, player_key sample: {df['player_key'].iloc[0]!r}")

    # Build player_id → team name from every match lineup
    player_team = {}
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    for match_id in matches["match_id"]:
        try:
            lineups = sb.lineups(match_id=match_id)
            for team_name, lineup_df in lineups.items():
                for pid in lineup_df["player_id"]:
                    player_team[pid] = team_name
        except Exception as e:
            print(f"  ⚠️  match {match_id} skipped: {e}")

    # Map — player_key may be int or string depending on StatsBomb version
    df["team"] = df["player_key"].map(player_team)
    # Fallback: try casting key to int if map returned all NaN
    if df["team"].isna().all():
        df["team"] = df["player_key"].astype(float).astype("Int64").map(player_team)

    filled = df["team"].notna().sum()
    print(f"  ✅ {filled}/{len(df)} players matched to a team")
    if filled == 0:
        print(f"  ❌ No matches — check player_key type vs lineup player_id type")
        print(f"     player_key dtype: {df['player_key'].dtype}, sample: {df['player_key'].iloc[:3].tolist()}")
        print(f"     lineup pid sample: {list(player_team.keys())[:3]}")
    else:
        df.to_parquet(path, index=False)
        print(f"  💾 Saved {path}")
