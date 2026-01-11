from __future__ import annotations
import pandas as pd
from statsbombpy import sb

from football_analytics.data.minutes import calc_minutes_played
from football_analytics.data.flatten import flatten_events

def get_competition_events(competition_id: int, season_id: int) -> pd.DataFrame:
    matches = sb.matches(competition_id=competition_id, season_id=season_id)

    all_events = []
    for match_id in matches["match_id"]:
        try:
            ev = sb.events(match_id=match_id)
            ev["match_id"] = match_id
            ev = calc_minutes_played(ev)
            ev = flatten_events(ev)
            all_events.append(ev)
        except Exception as e:
            print(f"⚠️ Skipped match {match_id}: {e}")

    if not all_events:
        return pd.DataFrame()

    df = pd.concat(all_events, ignore_index=True)

    key = "player_id" if "player_id" in df.columns else "player"
    mins_by_player = (
        df.drop_duplicates([key, "match_id"])
          .groupby(key)["total_mins_played"]
          .sum(min_count=1)
    )
    df["total_tourn_minutes"] = df[key].map(mins_by_player).fillna(0.0)

    return df
