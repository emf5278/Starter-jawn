"""Walk-forward calibration for the anytime-TD model.

    python -m pipeline.calibrate_td --season 2025 [--from-week 8]

For each week it rebuilds the model's inputs from *prior weeks only* — usage
shares, team splits, opponent factors — predicts P(>=1 TD) for every player
who was on a roster, and scores those predictions against who actually found
the end zone.  Reports:

  * Brier score and log loss vs. the base rate (a model that beats "everyone
    gets the base rate" has learned something)
  * a reliability table: predicted probability decile vs. observed hit rate,
    which is where you can see over- or under-confidence directly
  * top-20 hit rate, i.e. how the board's headline list would have done

Team expected TDs are held at the league average here rather than taken from
a game line, because historical closing totals are a paid Odds API feature.
That isolates the part this script is actually testing — how the model splits
a team's touchdowns among its players.
"""

from __future__ import annotations

import argparse
import logging
import math

import pandas as pd

from . import config
from .data import nfl
from .model import touchdowns as TD
from .run_touchdowns import _is_rookie

log = logging.getLogger("pipeline.td.calib")


def _week_predictions(season: int, week: int, usage_prior: dict,
                      roster: pd.DataFrame, scorers: dict) -> list[dict]:
    players_u = usage_prior["players"].groupby("gsis_id").sum(
        numeric_only=True).reset_index()
    teams_u = usage_prior["teams"]
    defence = usage_prior["defence"]

    sched = nfl.schedule(season)
    wk = sched[(sched["week"] == week) & (sched["game_type"] == "REG")]
    rows = []
    for _, g in wk.iterrows():
        gid = g["game_id"]
        if gid not in scorers:
            continue
        for team, opp in ((g["home_team"], g["away_team"]),
                          (g["away_team"], g["home_team"])):
            trow = teams_u[teams_u["posteam"] == team]
            trow = {} if trow.empty else trow.iloc[0].to_dict()
            drow = defence[defence["team"] == opp]
            drow = {} if drow.empty else drow.iloc[0].to_dict()

            team_off = TD.expected_team_tds(None)          # league average
            split = TD.team_rush_share(
                trow.get("rush_tds", 0.0),
                trow.get("rush_tds", 0.0) + trow.get("rec_tds", 0.0))
            opp_rush = TD.opponent_factor(drow.get("rush_tds_allowed", 0.0),
                                          drow.get("games", 0.0),
                                          config.TD_LEAGUE_RUSH_TD_PER_TEAM_GAME)
            opp_rec = TD.opponent_factor(drow.get("rec_tds_allowed", 0.0),
                                         drow.get("games", 0.0),
                                         config.TD_LEAGUE_REC_TD_PER_TEAM_GAME)

            squad = roster[roster["team"] == team]
            raw_rush, raw_rec, meta = {}, {}, {}
            for _, p in squad.iterrows():
                pid = p["gsis_id"]
                pos = str(p.get("position") or "")
                u = players_u[players_u["gsis_id"] == pid]
                u = {} if u.empty else u.iloc[0].to_dict()
                rookie = _is_rookie(p, season)
                rr, n_car = TD.raw_rush_share(u, trow, pos)
                re_, n_tgt = TD.raw_rec_share(u, trow)
                raw_rush[pid] = TD.shrink_share(rr, n_car, pos, p.get("draft_number"),
                                                rookie, "rush")
                raw_rec[pid] = 0.0 if pos == "QB" else TD.shrink_share(
                    re_, n_tgt, pos, p.get("draft_number"), rookie, "rec")
                meta[pid] = pos
            rs, cs = TD.normalise(raw_rush), TD.normalise(raw_rec)
            for pid in meta:
                pred = TD.predict_player(team_off, split, rs[pid], cs[pid],
                                         opp_rush, opp_rec)
                if pred["lambda"] < config.TD_MIN_LAMBDA_TO_LIST:
                    continue
                rows.append({"player_id": pid, "team": team, "week": week,
                             "prob": pred["prob"],
                             "scored": int(pid in scorers.get(gid, set()))})
    return rows


def run(season: int, from_week: int) -> None:
    sched = nfl.schedule(season)
    reg = sched[sched["game_type"] == "REG"]
    last = int(reg["week"].max())

    # who scored, per game, all season
    p = nfl._pbp(season)
    if p is None:
        raise SystemExit(f"no play-by-play for {season}")
    full = nfl._fetch_csv(
        nfl.PBP_URL.format(season=season), f"play_by_play_{season}.csv.gz",
        usecols=lambda c: c in ("game_id", "week", "season_type", "play_type",
                                "rusher_player_id", "receiver_player_id",
                                "rush_touchdown", "pass_touchdown"),
        low_memory=False, compression="gzip")
    full = full[full["season_type"] == "REG"]
    scorers: dict[str, set] = {}
    for gid in full["game_id"].unique():
        scorers[gid] = set()
    ru = full[(full["play_type"] == "run") & (full["rush_touchdown"] == 1)]
    for gid, pid in zip(ru["game_id"], ru["rusher_player_id"]):
        if isinstance(pid, str):
            scorers[gid].add(pid)
    pa = full[(full["play_type"] == "pass") & (full["pass_touchdown"] == 1)]
    for gid, pid in zip(pa["game_id"], pa["receiver_player_id"]):
        if isinstance(pid, str):
            scorers[gid].add(pid)

    roster_all = nfl.rosters(season)
    all_rows = []
    for week in range(from_week, last + 1):
        prior = nfl.usage(season, through_week=week - 1)
        if not prior:
            continue
        rost = nfl.rosters(season, week)
        if rost.empty:
            rost = roster_all
        rows = _week_predictions(season, week, prior, rost, scorers)
        all_rows.extend(rows)
        log.info("week %2d: %d player-predictions", week, len(rows))

    if not all_rows:
        raise SystemExit("no predictions produced")
    df = pd.DataFrame(all_rows)
    base = df["scored"].mean()
    brier = ((df["prob"] - df["scored"]) ** 2).mean()
    brier_base = ((base - df["scored"]) ** 2).mean()
    eps = 1e-9
    ll = -(df["scored"] * (df["prob"] + eps).apply(math.log)
           + (1 - df["scored"]) * (1 - df["prob"] + eps).apply(math.log)).mean()
    ll_base = -(df["scored"] * math.log(base)
                + (1 - df["scored"]) * math.log(1 - base)).mean()

    print(f"\nseason {season}, weeks {from_week}-{int(df.week.max())}")
    print(f"  predictions      : {len(df)}")
    print(f"  base rate        : {base:.4f}")
    print(f"  mean prediction  : {df['prob'].mean():.4f}")
    print(f"  Brier            : {brier:.5f}   (base {brier_base:.5f}, "
          f"skill {(1 - brier / brier_base) * 100:+.1f}%)")
    print(f"  log loss         : {ll:.5f}   (base {ll_base:.5f}, "
          f"skill {(1 - ll / ll_base) * 100:+.1f}%)")

    print("\n  reliability (predicted vs observed)")
    df["bin"] = pd.cut(df["prob"], [0, .05, .10, .15, .20, .25, .30, .40, .50, 1.0])
    tab = df.groupby("bin", observed=True).agg(
        n=("scored", "size"), predicted=("prob", "mean"), observed=("scored", "mean"))
    for b, r in tab.iterrows():
        flag = "" if abs(r.predicted - r.observed) < 0.03 else "  <-- off"
        print(f"    {str(b):>14}  n={int(r.n):5d}  pred {r.predicted:.3f}  "
              f"obs {r.observed:.3f}{flag}")

    print("\n  top-20 by probability, per week")
    hits = df.sort_values("prob", ascending=False).groupby("week").head(20)
    print(f"    picks {len(hits)}  hit rate {hits['scored'].mean():.3f}  "
          f"model expected {hits['prob'].mean():.3f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--from-week", type=int, default=8)
    args = ap.parse_args()
    run(args.season, args.from_week)


if __name__ == "__main__":
    main()
