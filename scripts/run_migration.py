import os
import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("ALTER TABLE player ADD COLUMN IF NOT EXISTS team TEXT")
conn.commit()
cur.close()
conn.close()
print("done — team column added")
