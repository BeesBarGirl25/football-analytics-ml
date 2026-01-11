import argparse
from pathlib import Path

from football_analytics.data.statsbomb import get_competition_events
from football_analytics.geometry.shared_cols import add_shared_geometry_cols
from football_analytics.features.player_level import build_player_level_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--competition-id", type=int, required=True)
    p.add_argument("--season-id", type=int, required=True)
    p.add_argument("--out", default="artifacts/features/player_level.parquet")
    args = p.parse_args()

    events = get_competition_events(args.competition_id, args.season_id)
    events = add_shared_geometry_cols(events)

    player_df = build_player_level_df(events)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    player_df.to_parquet(out, index=False)
    print(f"✅ wrote {len(player_df)} rows -> {out}")


if __name__ == "__main__":
    main()
