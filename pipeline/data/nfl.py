"""NFL data from nflverse (free, no API key) + the published schedule.

Sources
-------
* ``nfldata/data/games.csv``  — every game, past and scheduled, with kickoff
  time, teams, roof and (once played) the score.  The upcoming season's
  schedule is published here months before Week 1.
* ``nflverse-data`` releases — weekly rosters (who is on which team *now*)
  and play-by-play (usage: carries, targets, and where on the field they
  happened).

Everything is a plain CSV over HTTPS, cached on disk under ``data_cache/nfl``
so a re-run inside the same day costs nothing.  Play-by-play is ~19MB
gzipped per season, which is why the cache matters in CI.

Player ids are GSIS ids (``00-0034796``) throughout, which is what both the
rosters and the play-by-play use, so they join directly.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from zoneinfo import ZoneInfo

import pandas as pd

from .. import config

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
PBP_URL = RELEASE + "/pbp/play_by_play_{season}.csv.gz"
ROSTER_URL = RELEASE + "/weekly_rosters/roster_weekly_{season}.csv"

SKILL_POSITIONS = ("QB", "RB", "FB", "WR", "TE")
# Roster statuses that mean "on the active roster this week".  DEV is the
# practice squad and CUT/RES/RET are not playing.
ACTIVE_STATUSES = ("ACT",)

_CACHE_TTL_S = 6 * 3600


def _cache_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(os.path.dirname(root), "data_cache", "nfl")
    os.makedirs(d, exist_ok=True)
    return d


def _fetch_csv(url: str, name: str, ttl: int = _CACHE_TTL_S, **read_kw) -> pd.DataFrame | None:
    """Download a CSV to the on-disk cache and read it.

    Returns None (rather than raising) when the file does not exist yet —
    which is the normal state for the current season's play-by-play and
    stats before Week 1 has been played.
    """
    path = os.path.join(_cache_dir(), name)
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl
    if not fresh:
        import requests
        try:
            r = requests.get(url, timeout=180, stream=True)
            if r.status_code == 404:
                log.info("nflverse: %s not published yet (404)", name)
                return pd.read_csv(path, **read_kw) if os.path.exists(path) else None
            r.raise_for_status()
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            os.replace(tmp, path)
            log.info("nflverse: fetched %s (%.1f MB)", name, os.path.getsize(path) / 1048576)
        except Exception:
            log.warning("nflverse: fetch failed for %s", name, exc_info=True)
            if not os.path.exists(path):
                return None
            log.info("nflverse: falling back to cached %s", name)
    try:
        return pd.read_csv(path, **read_kw)
    except Exception:
        log.warning("nflverse: could not parse %s", name, exc_info=True)
        return None


# ---------------------------------------------------------------- schedule

def schedule(season: int) -> pd.DataFrame:
    df = _fetch_csv(GAMES_URL, "games.csv", low_memory=False)
    if df is None:
        raise SystemExit("could not load the NFL schedule")
    return df[df["season"] == season].copy()


def _kickoff_utc(gameday: str, gametime: str) -> dt.datetime | None:
    """games.csv carries local-to-ET kickoff times ('20:20')."""
    if not gameday or not isinstance(gametime, str) or ":" not in gametime:
        return None
    try:
        h, m = (int(x) for x in gametime.split(":")[:2])
        naive = dt.datetime.fromisoformat(str(gameday)).replace(hour=h, minute=m)
        return naive.replace(tzinfo=ET).astimezone(dt.timezone.utc)
    except Exception:
        return None


def games_on(date: dt.date, season: int | None = None) -> list[dict]:
    """Every game kicking off on `date` (US/Eastern), in kickoff order.

    An NFL "slate" is a calendar day, exactly like the MLB boards: a lone
    Thursday game, the big Sunday slate, Monday night.
    """
    season = season if season is not None else _season_for(date)
    sched = schedule(season)
    out = []
    for _, g in sched.iterrows():
        if str(g.get("gameday")) != date.isoformat():
            continue
        ko = _kickoff_utc(g.get("gameday"), g.get("gametime"))
        out.append({
            "game_id": g["game_id"],
            "season": int(g["season"]),
            "week": int(g["week"]),
            "game_type": g.get("game_type"),
            "kickoff_utc": ko.isoformat() if ko else None,
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "roof": g.get("roof"),
            "stadium": g.get("stadium"),
            "home_score": None if pd.isna(g.get("home_score")) else int(g["home_score"]),
            "away_score": None if pd.isna(g.get("away_score")) else int(g["away_score"]),
        })
    out.sort(key=lambda x: x["kickoff_utc"] or "")
    return out


def _season_for(date: dt.date) -> int:
    """NFL seasons straddle the new year: January games belong to the
    previous season's schedule."""
    return date.year - 1 if date.month < 3 else date.year


def current_week(date: dt.date, season: int | None = None) -> int | None:
    season = season if season is not None else _season_for(date)
    sched = schedule(season)
    played = sched[pd.to_datetime(sched["gameday"], errors="coerce").dt.date <= date]
    if played.empty:
        return int(sched["week"].min()) if not sched.empty else None
    return int(played["week"].max())


# ---------------------------------------------------------------- rosters

def rosters(season: int, week: int | None = None) -> pd.DataFrame:
    """Active skill-position players with their *current* team."""
    df = _fetch_csv(ROSTER_URL.format(season=season), f"roster_weekly_{season}.csv",
                    low_memory=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[df["position"].isin(SKILL_POSITIONS)]
    df = df[df["status"].isin(ACTIVE_STATUSES)]
    if week is not None and "week" in df.columns and not df.empty:
        weeks = sorted(df["week"].dropna().unique())
        pick = max([w for w in weeks if w <= week], default=weeks[-1] if weeks else None)
        if pick is not None:
            df = df[df["week"] == pick]
    keep = ["gsis_id", "full_name", "football_name", "position", "depth_chart_position",
            "team", "years_exp", "rookie_year", "entry_year", "draft_number", "week"]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")


# ------------------------------------------------------------ usage (pbp)

_PBP_COLS = ["season_type", "week", "posteam", "defteam", "play_type", "yardline_100",
             "rusher_player_id", "receiver_player_id", "rush_touchdown", "pass_touchdown"]


def _pbp(season: int) -> pd.DataFrame | None:
    df = _fetch_csv(PBP_URL.format(season=season), f"play_by_play_{season}.csv.gz",
                    usecols=lambda c: c in _PBP_COLS, low_memory=False,
                    compression="gzip")
    if df is None or df.empty:
        return None
    return df[df["season_type"] == "REG"]


def usage(season: int, through_week: int | None = None) -> dict:
    """Per-player rushing/receiving usage and the team totals to share out.

    Returns {"players": DataFrame, "teams": DataFrame, "games": DataFrame}
    where players carries, per (team, player):

        carries, goal_line_carries, rush_tds, targets, red_zone_targets, rec_tds

    and teams carries the same quantities summed over the team, which are the
    denominators the shares are taken against.
    """
    p = _pbp(season)
    if p is None:
        return {}
    if through_week is not None:
        p = p[p["week"] <= through_week]
    if p.empty:
        return {}

    gl = config.TD_GOAL_LINE_YARDLINE
    rz = config.TD_RED_ZONE_YARDLINE

    ru = p[(p["play_type"] == "run") & p["rusher_player_id"].notna()].copy()
    ru["goal_line"] = (ru["yardline_100"] <= gl).astype(int)
    rush = ru.groupby(["posteam", "rusher_player_id"]).agg(
        carries=("rush_touchdown", "size"),
        goal_line_carries=("goal_line", "sum"),
        rush_tds=("rush_touchdown", "sum"),
    ).reset_index().rename(columns={"rusher_player_id": "gsis_id"})

    pa = p[(p["play_type"] == "pass") & p["receiver_player_id"].notna()].copy()
    pa["red_zone"] = (pa["yardline_100"] <= rz).astype(int)
    rec = pa.groupby(["posteam", "receiver_player_id"]).agg(
        targets=("pass_touchdown", "size"),
        red_zone_targets=("red_zone", "sum"),
        rec_tds=("pass_touchdown", "sum"),
    ).reset_index().rename(columns={"receiver_player_id": "gsis_id"})

    players = rush.merge(rec, on=["posteam", "gsis_id"], how="outer").fillna(0)
    teams = players.groupby("posteam").sum(numeric_only=True).reset_index()

    # TDs allowed by each defence, for the opponent factor
    d_rush = ru.groupby("defteam")["rush_touchdown"].sum()
    d_rec = pa.groupby("defteam")["pass_touchdown"].sum()
    g = pd.concat([
        p.groupby("posteam")["week"].nunique().rename("games_off"),
        p.groupby("defteam")["week"].nunique().rename("games_def"),
    ], axis=1).fillna(0)
    defence = pd.DataFrame({
        "rush_tds_allowed": d_rush, "rec_tds_allowed": d_rec,
        "games": g["games_def"],
    }).fillna(0).reset_index().rename(columns={"index": "team"})

    return {"players": players, "teams": teams, "defence": defence,
            "team_games": g["games_off"].to_dict()}


def blended_usage(season: int, through_week: int | None = None) -> dict:
    """Current season blended with last season.

    Before Week 1 the current season's play-by-play does not exist at all, so
    this is simply last season — which is exactly the cold start the board
    badges as low confidence.
    """
    cur = usage(season, through_week)
    prev = usage(season - 1)
    if not cur and not prev:
        raise SystemExit(f"no play-by-play available for {season} or {season - 1}")
    if not cur:
        log.info("no %s play-by-play yet — running on %s usage alone", season, season - 1)
        return {**prev, "seasons_used": [season - 1], "current_season_games": 0}
    if not prev:
        return {**cur, "seasons_used": [season], "current_season_games": max(
            cur.get("team_games", {}).values(), default=0)}

    w = config.TD_PRIOR_SEASON_WEIGHT
    metrics = ["carries", "goal_line_carries", "rush_tds",
               "targets", "red_zone_targets", "rec_tds"]

    def _blend(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        m = a.merge(b, on=keys, how="outer", suffixes=("_cur", "_prev")).fillna(0)
        for c in metrics:
            m[c] = m[f"{c}_cur"] + w * m[f"{c}_prev"]
        return m[keys + metrics]

    players = _blend(cur["players"], prev["players"], ["posteam", "gsis_id"])
    # A player who changed teams should carry his usage to the new team; the
    # runner re-keys on the *current* roster, so collapse to player level too.
    by_player = players.groupby("gsis_id")[metrics].sum().reset_index()
    teams = players.groupby("posteam")[metrics].sum().reset_index()

    defence = cur["defence"].merge(prev["defence"], on="team", how="outer",
                                   suffixes=("_cur", "_prev")).fillna(0)
    for c in ("rush_tds_allowed", "rec_tds_allowed", "games"):
        defence[c] = defence[f"{c}_cur"] + w * defence[f"{c}_prev"]
    defence = defence[["team", "rush_tds_allowed", "rec_tds_allowed", "games"]]

    return {"players": players, "by_player": by_player, "teams": teams,
            "defence": defence, "seasons_used": [season, season - 1],
            "current_season_games": max(cur.get("team_games", {}).values(), default=0),
            "team_games": cur.get("team_games", {})}


def results_for(date: dt.date) -> dict[str, set[str]]:
    """{game_id: {gsis_id, ...}} — who actually scored a rushing or receiving
    TD in each game on `date`.  Used by the grader."""
    season = _season_for(date)
    p = _pbp(season)
    if p is None:
        return {}
    sched = schedule(season)
    ids = set(sched[sched["gameday"].astype(str) == date.isoformat()]["game_id"])
    if not ids:
        return {}
    # play-by-play carries game_id only when we ask for it; re-read narrowly
    df = _fetch_csv(PBP_URL.format(season=season), f"play_by_play_{season}.csv.gz",
                    usecols=lambda c: c in ("game_id", "play_type", "rusher_player_id",
                                            "receiver_player_id", "rush_touchdown",
                                            "pass_touchdown"),
                    low_memory=False, compression="gzip")
    if df is None:
        return {}
    df = df[df["game_id"].isin(ids)]
    out: dict[str, set[str]] = {gid: set() for gid in ids}
    ru = df[(df["play_type"] == "run") & (df["rush_touchdown"] == 1)]
    for gid, pid in zip(ru["game_id"], ru["rusher_player_id"]):
        if isinstance(pid, str):
            out.setdefault(gid, set()).add(pid)
    pa = df[(df["play_type"] == "pass") & (df["pass_touchdown"] == 1)]
    for gid, pid in zip(pa["game_id"], pa["receiver_player_id"]):
        if isinstance(pid, str):
            out.setdefault(gid, set()).add(pid)
    return out
