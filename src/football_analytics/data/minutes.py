from __future__ import annotations
import numpy as np
import pandas as pd

def calc_minutes_played(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    match_duration = int(df["minute"].max() or 0)
    df["total_mins_played"] = np.nan

    player_key = "player_id" if "player_id" in df.columns else "player"

    def _key(v):
        if isinstance(v, dict):
            return v.get("id") if player_key == "player_id" else v.get("name")
        return v

    for team, team_df in df.groupby("team", dropna=False):
        xi = team_df[team_df["type"] == "Starting XI"]
        if xi.empty:
            continue

        tactics = xi["tactics"].iloc[0] or {}
        lineup = tactics.get("lineup", [])

        intervals = {}
        for p in lineup:
            pinfo = p.get("player", {})
            pid = pinfo.get("id") if isinstance(pinfo, dict) else None
            pname = pinfo.get("name") if isinstance(pinfo, dict) else pinfo
            k = pid if player_key == "player_id" and pid is not None else pname
            intervals[k] = [(0, match_duration)]

        subs = team_df[team_df["type"] == "Substitution"][["minute", "player", "substitution_replacement"]].sort_values("minute")

        for _, r in subs.iterrows():
            m = int(r["minute"])
            off_k = _key(r["player"])
            on_k  = _key(r["substitution_replacement"])

            if off_k in intervals and intervals[off_k]:
                s, e = intervals[off_k][-1]
                intervals[off_k][-1] = (s, min(m, e))

            intervals.setdefault(on_k, []).append((m, match_duration))

        mins_map = {k: float(sum(max(0, b-a) for a, b in ivals)) for k, ivals in intervals.items()}

        mask = df["team"] == team
        df.loc[mask, "total_mins_played"] = df.loc[mask, player_key].map(mins_map)

    return df
