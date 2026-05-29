"""
Writes team names directly into the player table from StatsBomb lineup data.
Run via: heroku run python scripts/patch_team_in_db.py
"""
import os
import psycopg2
from statsbombpy import sb

MODEL_VERSION = os.getenv("MODEL_VERSION", "passing_v1")
DATABASE_URL = os.environ["DATABASE_URL"]

COMPETITIONS = [
    (43,   3,   "wc2018"),
    (43,   106, "wc2022"),
    (55,   43,  "euro2020"),
    (55,   282, "euro2024"),
    (1267, 107, "afcon2023"),
    (223,  282, "copa2024"),
]

# Build player_id (as str) → team name from all lineups
print("Fetching lineups from StatsBomb...")
player_team = {}
for comp_id, season_id, label in COMPETITIONS:
    try:
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        count = 0
        for match_id in matches["match_id"]:
            try:
                lineups = sb.lineups(match_id=match_id)
                for team_name, lineup_df in lineups.items():
                    for pid in lineup_df["player_id"]:
                        player_team[str(int(float(pid)))] = team_name
                count += 1
            except Exception as e:
                print(f"  ⚠️  {label} match {match_id}: {e}")
        print(f"  {label}: {count} matches processed")
    except Exception as e:
        print(f"  ❌ {label}: {e}")

print(f"\nTotal players with team: {len(player_team)}")
print(f"Sample: {list(player_team.items())[:5]}")

# Update DB
conn = psycopg2.connect(DATABASE_URL)
try:
    with conn.cursor() as cur:
        # Fetch all player rows for this model version
        cur.execute(
            "SELECT player_key, dataset FROM player WHERE model_version = %s",
            (MODEL_VERSION,),
        )
        rows = cur.fetchall()
        print(f"\nUpdating {len(rows)} player rows...")

        updated = 0
        for player_key, dataset in rows:
            key_str = str(int(float(player_key)))
            team = player_team.get(key_str)
            if team:
                cur.execute(
                    "UPDATE player SET team = %s WHERE model_version = %s AND player_key = %s AND dataset = %s",
                    (team, MODEL_VERSION, player_key, dataset),
                )
                updated += 1

        conn.commit()
        print(f"✅ Updated {updated}/{len(rows)} rows with team data")

        # Verify
        cur.execute(
            "SELECT DISTINCT team FROM player WHERE model_version = %s AND team IS NOT NULL AND team <> '' ORDER BY team LIMIT 10",
            (MODEL_VERSION,),
        )
        teams = [r[0] for r in cur.fetchall()]
        print(f"Teams in DB (first 10): {teams}")
finally:
    conn.close()
