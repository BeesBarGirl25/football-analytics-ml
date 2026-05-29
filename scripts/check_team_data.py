import os
import glob
import psycopg2
import pandas as pd

# Check parquet files
print("=== Parquet files ===")
for f in sorted(glob.glob("artifacts/features/player_level_*.parquet")):
    df = pd.read_parquet(f)
    has = "team" in df.columns
    if has and df["team"].notna().any():
        sample = df["team"].dropna().iloc[0]
    else:
        sample = "N/A"
    print(f"  {f.split('/')[-1].split(chr(92))[-1]:<40} team col: {has}  sample: {sample}")

# Check DB
print("\n=== Database ===")
conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("SELECT DISTINCT team FROM player WHERE model_version='passing_v1' AND team IS NOT NULL AND team <> '' LIMIT 20")
rows = cur.fetchall()
print(f"  Distinct teams (up to 20): {[r[0] for r in rows]}")

cur.execute("SELECT COUNT(*) FROM player WHERE model_version='passing_v1'")
print(f"  Total player rows: {cur.fetchone()[0]}")

cur.execute("SELECT COUNT(*) FROM player WHERE model_version='passing_v1' AND team IS NOT NULL AND team <> ''")
print(f"  Rows with team populated: {cur.fetchone()[0]}")

cur.close()
conn.close()
