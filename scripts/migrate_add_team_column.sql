-- Run once on the live Heroku DB to add the team column.
-- Safe to re-run (IF NOT EXISTS).
ALTER TABLE player ADD COLUMN IF NOT EXISTS team TEXT;
