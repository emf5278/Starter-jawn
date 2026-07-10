"""Opposing-bullpen inputs: season HR-vulnerability + recent usage (fatigue).

Two independent pieces, merged per team and keyed by MLB StatsAPI team id so
run.py can look up a batter's *opposing* bullpen directly:

  * HR-vulnerability — from the raw Statcast events already downloaded for
    pitcher splits.  A game's starter is the pitcher of that half-inning's
    first plate appearance; every other pitcher on that side is bullpen.  We
    aggregate relief HR / FB / BIP per pitching team.

  * Usage over the last 3 games — from StatsAPI box scores: relief batters
    faced (every pitcher with gamesStarted == 0) summed over the team's last
    three finals.  A heavily-worked pen leans on tired / lower-leverage arms.

Team abbreviations in Statcast (AZ, CWS, KC, SD, SF, TB, WSH, ATH, …) match
the `team` field in pipeline.stadiums, so we invert that table to go from the
Statcast abbrev to a StatsAPI team id.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
import requests

from ..stadiums import STADIUMS

log = logging.getLogger(__name__)

BASE = "https://statsapi.mlb.com/api/v1"
_ABBREV_TO_ID = {v["team"]: tid for tid, v in STADIUMS.items()}


def bullpen_hr_by_team(events: pd.DataFrame) -> dict[int, dict]:
    """team_id -> {"hr","fb","bip","pa"} for that team's relievers, season YTD."""
    ev = events[events["events"].notna() & (events["events"] != "")].copy()
    if ev.empty:
        return {}
    ev["pitch_abbrev"] = np.where(
        ev["inning_topbot"] == "Top", ev["home_team"], ev["away_team"])
    # starter = pitcher of each half-inning's first plate appearance
    starters = ev.loc[
        ev.groupby(["game_pk", "inning_topbot"])["at_bat_number"].idxmin(),
        ["game_pk", "inning_topbot", "pitcher"],
    ].rename(columns={"pitcher": "starter"})
    ev = ev.merge(starters, on=["game_pk", "inning_topbot"], how="left")
    pen = ev[ev["pitcher"] != ev["starter"]].copy()

    pen["hr"] = (pen["events"] == "home_run").astype(int)
    pen["fb"] = pen["bb_type"].isin(["fly_ball", "popup"]).astype(int)
    pen["bip"] = pen["bb_type"].notna().astype(int)
    g = pen.groupby("pitch_abbrev").agg(
        hr=("hr", "sum"), fb=("fb", "sum"), bip=("bip", "sum"), pa=("hr", "size"))

    out: dict[int, dict] = {}
    for abbrev, row in g.iterrows():
        tid = _ABBREV_TO_ID.get(abbrev)
        if tid is not None:
            out[tid] = {k: int(row[k]) for k in ("hr", "fb", "bip", "pa")}
    return out


def _relief_bf_for_game(game_pk: int, team_id: int) -> int | None:
    """Relief batters faced by `team_id` in one game (pitchers, GS == 0)."""
    try:
        box = requests.get(f"{BASE}/game/{game_pk}/boxscore", timeout=20).json()
        for side in ("home", "away"):
            if box["teams"][side]["team"]["id"] != team_id:
                continue
            total = 0
            for pl in box["teams"][side]["players"].values():
                pit = pl.get("stats", {}).get("pitching", {})
                if not pit:
                    continue
                gs = pl.get("seasonStats", {}).get("pitching", {}).get("gamesStarted")
                # per-game start flag: a starter pitched the 1st inning; simplest
                # reliable signal is battersFaced present AND not the game starter
                if pit.get("gamesStarted", 0) == 0:
                    total += int(pit.get("battersFaced", 0) or 0)
            return total
    except Exception:
        log.debug("relief BF lookup failed for game %s team %s", game_pk, team_id, exc_info=True)
    return None


def bullpen_usage_last3(team_id: int, before: dt.date) -> int | None:
    """Relief batters faced by a team over its last 3 completed games."""
    start = (before - dt.timedelta(days=8)).isoformat()
    end = (before - dt.timedelta(days=1)).isoformat()
    try:
        sched = requests.get(
            f"{BASE}/schedule",
            params={"sportId": 1, "teamId": team_id, "startDate": start, "endDate": end},
            timeout=20,
        ).json()
        finals = [
            g["gamePk"]
            for d in sched.get("dates", [])
            for g in d.get("games", [])
            if g.get("status", {}).get("abstractGameState") == "Final"
        ][-3:]
    except Exception:
        log.debug("schedule lookup failed for team %s", team_id, exc_info=True)
        return None
    if not finals:
        return None
    vals = [_relief_bf_for_game(pk, team_id) for pk in finals]
    vals = [v for v in vals if v is not None]
    return int(sum(vals)) if vals else None


def opposing_bullpens(events, team_ids: set[int], date: dt.date) -> dict[int, dict]:
    """team_id -> {"hr","fb","bip","bf_last3"} for every team on the slate."""
    hr = bullpen_hr_by_team(events) if events is not None else {}
    out: dict[int, dict] = {}
    for tid in team_ids:
        entry = dict(hr.get(tid, {}))
        entry["bf_last3"] = bullpen_usage_last3(tid, date)
        out[tid] = entry
    n_hr = sum(1 for v in out.values() if v.get("fb"))
    n_use = sum(1 for v in out.values() if v.get("bf_last3") is not None)
    log.info("bullpen: %d teams with season HR data, %d with 3-game usage",
             n_hr, n_use)
    return out
