"""Moneyline + totals board entry point (separate from HR and strikeout runs).

    python -m pipeline.run_gamelines [--date YYYY-MM-DD]
                                     [--output web/gamelines.json] [--no-odds]

For every game on today's slate: expected runs per team (offense, opposing
starter+bullpen FIP factors, park runs environment, home field), win
probabilities and a full total-runs distribution, then a comparison against the
de-vigged market moneyline and total.  Writes web/gamelines.json for the
Moneylines and Totals tabs. Does not touch predictions.json or strikeouts.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from . import config
from .data import lineups, odds as odds_mod, statcast
from .model import gamelines as G
from .stadiums import stadium_for_home_team

log = logging.getLogger("pipeline.gl")

MODEL_VERSION = "gl-0.1.0"
ET = ZoneInfo("America/New_York")

# StatsAPI team id -> Savant abbrev (invert the stadiums table)
_ID_TO_ABBREV = {}


def _abbrev(team_id: int) -> str | None:
    if not _ID_TO_ABBREV:
        from .stadiums import STADIUMS
        _ID_TO_ABBREV.update({tid: v["team"] for tid, v in STADIUMS.items()})
    return _ID_TO_ABBREV.get(team_id)


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _ml_ev(p_win: float, price: float) -> float:
    return p_win * (price - 1) - (1 - p_win)


def _total_ev(p_win: float, p_push: float, price: float) -> float:
    """Push returns the stake: EV = p_win*(d-1) - p_lose."""
    p_lose = max(0.0, 1.0 - p_win - p_push)
    return p_win * (price - 1) - p_lose


def run(date: dt.date, output: str, use_odds: bool) -> dict:
    season = date.year
    log.info("fetching slate for %s", date)
    slate = lineups.todays_slate(date)

    log.info("fetching raw statcast events (cached)")
    events = statcast.season_events(season, date)
    if events is None:
        raise SystemExit("no Statcast events; cannot build game lines")

    league_rpg = statcast.league_rpg_from_events(events)
    offense = statcast.team_offense_from_events(events)
    parks = statcast.park_runs_from_events(events)
    fip = statcast.pitcher_fip_inputs_from_events(events)
    pens = statcast.bullpen_fip_by_team_from_events(events)
    lg_fip_pa = float((13 * fip["hr"].sum() + 3 * fip["bb"].sum()
                       - 2 * fip["k"].sum()) / fip["pa"].sum())
    league = {"rpg": league_rpg, "fip_pa": lg_fip_pa}
    log.info("league RPG=%.2f fip_pa=%.3f | %d teams, %d parks",
             league_rpg, lg_fip_pa, len(offense), len(parks))

    market = []
    if use_odds and os.environ.get("ODDS_API_KEY"):
        log.info("fetching moneyline + totals odds")
        market = odds_mod.fetch_main_market_odds(
            os.environ["ODDS_API_KEY"], os.environ.get("ODDS_REGIONS", "us"))
    elif use_odds:
        log.warning("ODDS_API_KEY not set; skipping market comparison")
    mkt_by_home = {m["home_norm"]: m for m in market}

    def team_row(abbrev: str | None) -> dict:
        if abbrev and abbrev in offense.index:
            r = offense.loc[abbrev]
            return G.offense_factor(float(r["rpg"]), float(r["games"]), league_rpg)
        return G.offense_factor(None, None, league_rpg)

    def starter_stats(pid: int | None) -> dict | None:
        if pid and pid in fip.index:
            r = fip.loc[pid]
            return {k: float(r[k]) for k in ("pa", "k", "bb", "hr")}
        return None

    def pen_stats(abbrev: str | None) -> dict | None:
        if abbrev and abbrev in pens.index:
            r = pens.loc[abbrev]
            return {k: float(r[k]) for k in ("pa", "k", "bb", "hr")}
        return None

    games_out = []
    for game in slate["games"]:
        home, away = game["teams"]["home"], game["teams"]["away"]
        h_ab, a_ab = _abbrev(home["team_id"]), _abbrev(away["team_id"])
        stadium = stadium_for_home_team(home["team_id"])

        off_h, off_a = team_row(h_ab), team_row(a_ab)
        st_vs_h = G.pitching_factor(starter_stats(away["probable_pitcher_id"]), league,
                                    config.GL_STARTER_BALLAST_PA, config.GL_STARTER_FACTOR_CAP)
        st_vs_a = G.pitching_factor(starter_stats(home["probable_pitcher_id"]), league,
                                    config.GL_STARTER_BALLAST_PA, config.GL_STARTER_FACTOR_CAP)
        pen_vs_h = G.pitching_factor(pen_stats(a_ab), league,
                                     config.GL_BULLPEN_BALLAST_PA, config.GL_BULLPEN_FACTOR_CAP)
        pen_vs_a = G.pitching_factor(pen_stats(h_ab), league,
                                     config.GL_BULLPEN_BALLAST_PA, config.GL_BULLPEN_FACTOR_CAP)
        if h_ab and h_ab in parks.index:
            pr = parks.loc[h_ab]
            park = G.park_runs_factor(float(pr["total_rpg"]), float(pr["games"]), league_rpg)
        else:
            park = G.park_runs_factor(None, None, league_rpg)

        lam_h = G.team_lambda(league_rpg, off_h, st_vs_h, pen_vs_h, park, True)
        lam_a = G.team_lambda(league_rpg, off_a, st_vs_a, pen_vs_a, park, False)
        wp = G.win_probability(lam_h, lam_a)
        expected_total = round(lam_h + lam_a, 2)

        row = {
            "game_pk": game["game_pk"],
            "game_time_utc": game["game_time_utc"],
            "venue": stadium["name"] if stadium else game.get("venue"),
            "home": {"team": home["team_name"], "pitcher": home["probable_pitcher_name"],
                     "lambda": round(lam_h, 2), "p_win": wp["p_home"]},
            "away": {"team": away["team_name"], "pitcher": away["probable_pitcher_name"],
                     "lambda": round(lam_a, 2), "p_win": wp["p_away"]},
            "expected_total": expected_total,
            "factors": {
                "league_rpg": round(league_rpg, 2),
                "offense_home": off_h, "offense_away": off_a,
                "starter_vs_home": st_vs_h, "starter_vs_away": st_vs_a,
                "bullpen_vs_home": pen_vs_h, "bullpen_vs_away": pen_vs_a,
                "park": {**park, "name": stadium["name"] if stadium else None},
                "home_boost": config.GL_HOME_BOOST,
            },
            "moneyline": None, "total": None,
        }

        mkt = mkt_by_home.get(odds_mod.normalize_name(home["team_name"]))
        if mkt and mkt.get("moneyline"):
            ml = mkt["moneyline"]
            ev_h = _ml_ev(wp["p_home"], ml["home_price_decimal"])
            ev_a = _ml_ev(wp["p_away"], ml["away_price_decimal"])
            side = "home" if ev_h >= ev_a else "away"
            row["moneyline"] = {
                **ml, "fair_home": round(ml["fair_home"], 4),
                "ev_home": round(ev_h, 4), "ev_away": round(ev_a, 4),
                "best_side": side, "best_ev": round(max(ev_h, ev_a), 4),
                "edge_home": round(wp["p_home"] - ml["fair_home"], 4),
            }
        if mkt and mkt.get("total"):
            tt = mkt["total"]
            tp = G.total_probs(tt["line"], lam_h, lam_a)
            ev_o = _total_ev(tp["p_over"], tp["p_push"], tt["over_price_decimal"])
            ev_u = _total_ev(tp["p_under"], tp["p_push"], tt["under_price_decimal"])
            side = "Over" if ev_o >= ev_u else "Under"
            row["total"] = {
                **tt, "fair_over": round(tt["fair_over"], 4), **tp,
                "ev_over": round(ev_o, 4), "ev_under": round(ev_u, 4),
                "best_side": side, "best_ev": round(max(ev_o, ev_u), 4),
                "edge_over": round(tp["p_over"] / max(1e-9, 1 - tp["p_push"])
                                   - tt["fair_over"], 4),
            }
        games_out.append(row)

    games_out.sort(key=lambda r: r["game_time_utc"] or "")
    doc = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds"),
        "date": date.isoformat(),
        "model_version": MODEL_VERSION,
        "league_rpg": round(league_rpg, 2),
        "n_games": len(games_out),
        "odds_available": bool(market),
        "games": games_out,
    }
    safe = _json_safe(doc)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(safe, f, indent=2, allow_nan=False)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hist = os.path.join(root, "history", "gamelines")
    os.makedirs(hist, exist_ok=True)
    with open(os.path.join(hist, f"{date.isoformat()}.json"), "w") as f:
        json.dump(safe, f, indent=2, allow_nan=False)
    log.info("wrote %s (%d games)", output, len(games_out))
    return doc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    try:
        from pybaseball import cache
        cache.enable()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date")
    ap.add_argument("--output", default="web/gamelines.json")
    ap.add_argument("--no-odds", action="store_true")
    args = ap.parse_args()
    date = (dt.date.fromisoformat(args.date) if args.date
            else dt.datetime.now(ET).date())
    run(date, args.output, not args.no_odds)


if __name__ == "__main__":
    main()
