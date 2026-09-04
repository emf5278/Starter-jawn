"""NFL anytime-TD board entry point.

    python -m pipeline.run_touchdowns [--date YYYY-MM-DD]
                                      [--output web/touchdowns.json] [--no-odds]

Projects P(>=1 rushing or receiving TD) for every active skill player in the
day's games, compares it to the anytime-TD price, and writes the JSON the
touchdowns.html page renders.  Mirrors the HR board: top 20 by probability,
top 20 by EV, union of the two published.

An NFL slate is a calendar day, so most days of the week there is nothing to
do.  On those days this exits immediately, before touching The Odds API — the
whole point of the cheap schedule.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from . import config
from .data import nfl, odds as odds_mod
from .model import touchdowns as TD
from .model.predict import ev_per_dollar

log = logging.getLogger("pipeline.td")

MODEL_VERSION = "td-0.1.0"
ET = ZoneInfo("America/New_York")


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if o is None or isinstance(o, (str, int, bool)):
        return o
    if isinstance(o, float):
        return o
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if math.isfinite(v) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
    except Exception:
        pass
    return o


def _row(df: pd.DataFrame, key: str, value) -> dict:
    hit = df[df[key] == value]
    return {} if hit.empty else hit.iloc[0].to_dict()


def run(date: dt.date, output: str, use_odds: bool) -> dict:
    season = nfl._season_for(date)
    games = nfl.games_on(date, season)
    if not games:
        log.info("no NFL games on %s — nothing to do", date)
        return {"date": date.isoformat(), "games": 0, "skipped": True}

    week = int(games[0]["week"])
    log.info("%s: %d game(s), season %d week %d", date, len(games), season, week)

    use = nfl.blended_usage(season, through_week=week - 1)
    players_u = use.get("by_player")
    if players_u is None:
        players_u = use["players"].groupby("gsis_id").sum(numeric_only=True).reset_index()
    teams_u, defence = use["teams"], use["defence"]
    cur_games = use.get("current_season_games", 0)
    log.info("usage from seasons %s | current-season games so far: %s",
             use.get("seasons_used"), cur_games)

    roster = nfl.rosters(season, week)
    if roster.empty:
        raise SystemExit(f"no {season} roster data available")
    log.info("active skill players on rosters: %d", len(roster))

    # last season's team for each player, to flag off-season moves
    prev_team = {}
    if "players" in use and "posteam" in use["players"].columns:
        pv = use["players"].sort_values("carries", ascending=False)
        for gid, tm in zip(pv["gsis_id"], pv["posteam"]):
            prev_team.setdefault(gid, tm)

    game_lines = {}
    td_props = {}
    if use_odds and os.environ.get("ODDS_API_KEY"):
        key = os.environ["ODDS_API_KEY"]
        regions = os.environ.get("ODDS_REGIONS", "us")
        log.info("fetching NFL game lines (totals/spreads)")
        game_lines = odds_mod.fetch_game_lines(
            key, regions, sport=config.ODDS_NFL_SPORT_KEY,
            markets=config.ODDS_NFL_GAME_MARKETS)
        log.info("fetching anytime-TD props")
        td_props = odds_mod.fetch_anytime_td_props(key, regions, on_date=date)
    elif use_odds:
        log.warning("ODDS_API_KEY not set; skipping odds/EV and game lines")

    league_rush_pg = config.TD_LEAGUE_RUSH_TD_PER_TEAM_GAME
    league_rec_pg = config.TD_LEAGUE_REC_TD_PER_TEAM_GAME

    rows: list[dict] = []
    for game in games:
        for side, opp_side in (("home", "away"), ("away", "home")):
            team = game[f"{side}_team"]
            opp = game[f"{opp_side}_team"]

            line = game_lines.get(odds_mod.normalize_name(_full_team_name(team)), {})
            implied = line.get("implied_team_runs")
            team_off = TD.expected_team_tds(implied)

            trow = _row(teams_u, "posteam", team)
            split = TD.team_rush_share(trow.get("rush_tds", 0.0),
                                       trow.get("rush_tds", 0.0) + trow.get("rec_tds", 0.0))

            drow = _row(defence, "team", opp)
            opp_rush = TD.opponent_factor(drow.get("rush_tds_allowed", 0.0),
                                          drow.get("games", 0.0), league_rush_pg)
            opp_rec = TD.opponent_factor(drow.get("rec_tds_allowed", 0.0),
                                         drow.get("games", 0.0), league_rec_pg)

            squad = roster[roster["team"] == team]
            if squad.empty:
                log.warning("no roster rows for %s", team)
                continue

            # ---- raw shares for everyone on the roster, then normalise ----
            raw_rush: dict[str, float] = {}
            raw_rec: dict[str, float] = {}
            meta: dict[str, dict] = {}
            for _, p in squad.iterrows():
                gid = p["gsis_id"]
                pos = str(p.get("position") or "")
                urow = _row(players_u, "gsis_id", gid)
                rookie = _is_rookie(p, season)

                rr, n_car = TD.raw_rush_share(urow, trow, pos)
                re_, n_tgt = TD.raw_rec_share(urow, trow)
                raw_rush[gid] = TD.shrink_share(rr, n_car, pos, p.get("draft_number"),
                                                rookie, "rush")
                # Passing TDs never count, so a QB is carried on rushing only.
                raw_rec[gid] = 0.0 if pos == "QB" else TD.shrink_share(
                    re_, n_tgt, pos, p.get("draft_number"), rookie, "rec")
                meta[gid] = {
                    "name": p.get("full_name") or p.get("football_name") or gid,
                    "position": pos,
                    "rookie": rookie,
                    "changed_teams": bool(prev_team.get(gid) and prev_team[gid] != team),
                    "carries": float(urow.get("carries", 0) or 0),
                    "targets": float(urow.get("targets", 0) or 0),
                    "goal_line_carries": float(urow.get("goal_line_carries", 0) or 0),
                    "red_zone_targets": float(urow.get("red_zone_targets", 0) or 0),
                    "prior_rush_tds": float(urow.get("rush_tds", 0) or 0),
                    "prior_rec_tds": float(urow.get("rec_tds", 0) or 0),
                }

            rush_shares = TD.normalise(raw_rush)
            rec_shares = TD.normalise(raw_rec)

            for gid, m in meta.items():
                pred = TD.predict_player(team_off, split, rush_shares[gid],
                                         rec_shares[gid], opp_rush, opp_rec)
                if pred["lambda"] < config.TD_MIN_LAMBDA_TO_LIST:
                    continue
                conf = TD.confidence(cur_games, m["changed_teams"], m["rookie"])
                row = {
                    "player_id": gid,
                    "name": m["name"],
                    "position": m["position"],
                    "team": team,
                    "opponent": opp,
                    "home": side == "home",
                    "game_id": game["game_id"],
                    "kickoff_utc": game["kickoff_utc"],
                    "week": game["week"],
                    "confidence": conf,
                    "rookie": m["rookie"],
                    "changed_teams": m["changed_teams"],
                    **pred,
                    "shares": {
                        "rush": round(rush_shares[gid], 4),
                        "rec": round(rec_shares[gid], 4),
                    },
                    "usage": {
                        "carries": m["carries"], "goal_line_carries": m["goal_line_carries"],
                        "targets": m["targets"], "red_zone_targets": m["red_zone_targets"],
                        "prior_rush_tds": m["prior_rush_tds"],
                        "prior_rec_tds": m["prior_rec_tds"],
                    },
                    "factors": {
                        "team_off_tds": team_off,
                        "team_rush_split": split,
                        "opp_rush": opp_rush,
                        "opp_rec": opp_rec,
                        "game_line": line or None,
                    },
                }
                # Book-independent: the shortest price at which this is +EV.
                row["break_even_american"] = (
                    odds_mod.decimal_to_american(1.0 / pred["prob"])
                    if pred["prob"] > 0 else None)
                rows.append(row)

    # ---- attach odds / EV -------------------------------------------------
    for r in rows:
        q = td_props.get(odds_mod.normalize_name(r["name"]))
        if q:
            ev = ev_per_dollar(r["prob"], q["best_price_decimal"])
            r["odds"] = {
                "best_price_decimal": q["best_price_decimal"],
                "best_price_american": q["best_price_american"],
                "best_book": q["best_book"],
                "implied_prob": round(q["implied_prob"], 4),
                "fair_prob": round(q["fair_prob"], 4),
                "n_books": q["n_books"],
                "two_sided": q.get("two_sided", False),
                "ev_per_dollar": round(ev, 4),
                "edge_vs_fair": round(r["prob"] - q["fair_prob"], 4),
            }
        else:
            r["odds"] = None

    rows.sort(key=lambda r: r["prob"], reverse=True)
    top_prob = rows[:config.TD_TOP_N]
    top_ev = sorted((r for r in rows if r["odds"]),
                    key=lambda r: r["odds"]["ev_per_dollar"], reverse=True)[:config.TD_TOP_N]
    seen, top = set(), []
    for r in top_prob + top_ev:
        if r["player_id"] not in seen:
            seen.add(r["player_id"])
            top.append(r)
    top.sort(key=lambda r: r["prob"], reverse=True)

    doc = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds"),
        "date": date.isoformat(),
        "season": season,
        "week": week,
        "model_version": MODEL_VERSION,
        "n_games": len(games),
        "n_players_scored": len(rows),
        "odds_available": bool(td_props),
        "game_lines_available": bool(game_lines),
        "seasons_used": use.get("seasons_used"),
        "current_season_games": cur_games,
        "cold_start": cur_games < config.TD_CONF_MEDIUM_MIN_GAMES,
        "league": {
            "rush_td_per_team_game": league_rush_pg,
            "rec_td_per_team_game": league_rec_pg,
        },
        "top_n": config.TD_TOP_N,
        "games": games,
        "players": top,
    }
    safe = _json_safe(doc)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(safe, f, indent=2, allow_nan=False)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hist = os.path.join(root, "history", "touchdowns")
    os.makedirs(hist, exist_ok=True)
    with open(os.path.join(hist, f"{date.isoformat()}.json"), "w") as f:
        json.dump(safe, f, indent=2, allow_nan=False)
    log.info("wrote %s (%d players scored, top %d kept, odds=%s)",
             output, len(rows), len(top), bool(td_props))
    return doc


# nflverse uses abbreviations (KC); The Odds API uses full names (Kansas City
# Chiefs).  One small table beats a fuzzy match that silently mispairs teams.
TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints",
    "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def _full_team_name(abbr: str) -> str:
    return TEAM_NAMES.get(abbr, abbr)


def _is_rookie(p, season: int) -> bool:
    for col in ("rookie_year", "entry_year"):
        v = p.get(col)
        try:
            if v is not None and int(v) == season:
                return True
        except (TypeError, ValueError):
            continue
    try:
        return float(p.get("years_exp")) == 0
    except (TypeError, ValueError):
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="YYYY-MM-DD (default: today in US/Eastern)")
    ap.add_argument("--output", default="web/touchdowns.json")
    ap.add_argument("--no-odds", action="store_true")
    args = ap.parse_args()
    date = (dt.date.fromisoformat(args.date) if args.date
            else dt.datetime.now(ET).date())
    run(date, args.output, not args.no_odds)


if __name__ == "__main__":
    main()
