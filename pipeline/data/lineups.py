"""Today's slate from the MLB StatsAPI (statsapi.mlb.com, public, no key).

For each game we return the probable pitchers and, when posted, the confirmed
batting orders.  If a team hasn't posted its lineup yet (common at 9am ET),
we fall back to that team's most recent completed-game batting order and flag
it `lineup_confirmed: False`.
"""

from __future__ import annotations

import datetime as dt
import logging

import requests

log = logging.getLogger(__name__)

BASE = "https://statsapi.mlb.com/api/v1"


def _get(path: str, **params) -> dict:
    r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _player_details(person_ids: list[int]) -> dict[int, dict]:
    """Batch lookup: bat side / pitch hand / full name per player id."""
    out: dict[int, dict] = {}
    for i in range(0, len(person_ids), 100):
        chunk = person_ids[i : i + 100]
        data = _get("people", personIds=",".join(map(str, chunk)))
        for p in data.get("people", []):
            out[p["id"]] = {
                "name": p.get("fullName", ""),
                "bat_side": p.get("batSide", {}).get("code", "R"),
                "pitch_hand": p.get("pitchHand", {}).get("code", "R"),
            }
    return out


def _last_lineup(team_id: int, before: dt.date) -> list[int]:
    """Batting order (list of 9 player ids) from the team's last final game."""
    start = (before - dt.timedelta(days=10)).strftime("%Y-%m-%d")
    end = (before - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        sched = _get("schedule", sportId=1, teamId=team_id, startDate=start, endDate=end)
        games = [
            g
            for d in sched.get("dates", [])
            for g in d.get("games", [])
            if g.get("status", {}).get("abstractGameState") == "Final"
        ]
        if not games:
            return []
        last = games[-1]
        box = _get(f"game/{last['gamePk']}/boxscore")
        side = "home" if last["teams"]["home"]["team"]["id"] == team_id else "away"
        order = box["teams"][side].get("battingOrder", [])
        return [int(pid) for pid in order[:9]]
    except Exception:
        log.warning("fallback lineup lookup failed for team %s", team_id, exc_info=True)
        return []


def todays_slate(date: dt.date) -> dict:
    """{"games": [...], "players": {id: {name, bat_side, pitch_hand}}}."""
    sched = _get(
        "schedule",
        sportId=1,
        date=date.strftime("%Y-%m-%d"),
        hydrate="probablePitcher,lineups",
    )
    games = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                continue
            entry = {
                "game_pk": g["gamePk"],
                "game_time_utc": g.get("gameDate"),
                "venue": g.get("venue", {}).get("name"),
                "teams": {},
            }
            lineup_key = {"home": "homePlayers", "away": "awayPlayers"}
            for side in ("home", "away"):
                t = g["teams"][side]
                posted = [
                    p["id"] for p in g.get("lineups", {}).get(lineup_key[side], [])
                ][:9]
                confirmed = len(posted) == 9
                lineup = posted if confirmed else _last_lineup(t["team"]["id"], date)
                prob = t.get("probablePitcher") or {}
                entry["teams"][side] = {
                    "team_id": t["team"]["id"],
                    "team_name": t["team"].get("name", ""),
                    "probable_pitcher_id": prob.get("id"),
                    "probable_pitcher_name": prob.get("fullName"),
                    "lineup": lineup,
                    "lineup_confirmed": confirmed,
                }
            games.append(entry)

    # one batched handedness/name lookup for everyone on the slate
    ids: set[int] = set()
    for g in games:
        for side in ("home", "away"):
            ids.update(g["teams"][side]["lineup"])
            if g["teams"][side]["probable_pitcher_id"]:
                ids.add(g["teams"][side]["probable_pitcher_id"])
    details = _player_details(sorted(ids))
    return {"games": games, "players": details}
