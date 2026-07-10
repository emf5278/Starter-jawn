"""Daily pipeline entry point.

    python -m pipeline.run [--date YYYY-MM-DD] [--output web/predictions.json]
                           [--lite] [--no-odds] [--top 10]

Steps: slate + lineups (StatsAPI) -> batter/pitcher stats (pybaseball) ->
weather (Open-Meteo) -> model -> HR prop odds (The Odds API) -> JSON.

--lite skips the raw-Statcast pitcher handedness splits (slow on a cold
cache) and uses FanGraphs overall pitcher rates instead.
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
from .data import lineups, odds as odds_mod, statcast, weather as weather_mod
from .model.predict import ev_per_dollar, predict_player
from .stadiums import stadium_for_home_team

log = logging.getLogger("pipeline")

MODEL_VERSION = "0.1.0"


def _json_safe(o):
    """Replace non-finite floats (NaN/inf) with None, recursively."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _batter_hand(bat_side: str, pitcher_hand: str | None) -> str:
    if bat_side == "S":  # switch hitter bats opposite the pitcher
        return "L" if (pitcher_hand or "R") == "R" else "R"
    return bat_side or "R"


def _pitcher_split(splits, overall, pitcher_id: int | None, stand: str) -> dict | None:
    if pitcher_id is None:
        return None
    if splits is not None:
        try:
            row = splits.loc[(pitcher_id, stand)]
            return {k: float(row[k]) for k in ("pa", "hr", "fb", "gb", "bip")}
        except KeyError:
            pass
    if overall is not None and pitcher_id in overall.index:
        row = overall.loc[pitcher_id]
        return {"hr_fb": row["hr_fb"], "fb_pct": row["fb_pct"], "TBF": row["TBF"]}
    return None


def run(date: dt.date, output: str, lite: bool, use_odds: bool, top_n: int) -> dict:
    season = date.year

    log.info("fetching slate + lineups for %s", date)
    slate = lineups.todays_slate(date)
    players = slate["players"]

    log.info("fetching batter power stats (%s + %s blend)", season, season - 1)
    batters = statcast.batter_power_stats(season)
    league = statcast.league_baselines(batters, season)

    splits = None
    if not lite:
        log.info("fetching raw statcast events (cached, incremental)")
        events = statcast.season_events(season, date)
        if events is not None:
            splits = statcast.pitcher_splits_from_events(events)
            # league rates straight from events beat the FanGraphs-dependent
            # estimates (FanGraphs 403s from cloud IPs)
            league.update(statcast.league_rates_from_events(events))
            # fill HR/FB for batters FanGraphs couldn't provide
            hrfb = statcast.batter_hrfb_from_events(events)
            batters = batters.join(
                hrfb.rename(columns={"hr_fb": "hr_fb_ev", "fb": "fb_ev"}), how="left")
            batters["hr_fb"] = batters["hr_fb"].fillna(batters["hr_fb_ev"])
            # a 0 sample (FanGraphs absent) must also count as missing, or the
            # regression treats every hitter as league-average on HR/FB
            batters["hr_fb_n"] = (
                batters["hr_fb_n"].mask(batters["hr_fb_n"] <= 0).fillna(batters["fb_ev"])
            )
    log.info("league baselines: %s", {k: round(v, 4) for k, v in league.items()})
    overall = None
    try:
        overall = statcast.pitcher_overall_stats(season)
    except Exception:
        log.warning("FanGraphs pitcher stats unavailable", exc_info=True)

    prop_odds = {}
    if use_odds:
        key = os.environ.get("ODDS_API_KEY")
        if key:
            log.info("fetching HR prop odds")
            prop_odds = odds_mod.fetch_hr_props(key, os.environ.get("ODDS_REGIONS", "us"))
            log.info("odds found for %d players", len(prop_odds))
        else:
            log.warning("ODDS_API_KEY not set; skipping odds/EV")

    rows = []
    for game in slate["games"]:
        stadium = stadium_for_home_team(game["teams"]["home"]["team_id"])
        if stadium is None:
            log.warning("unknown venue for game %s; skipping", game["game_pk"])
            continue
        wx = weather_mod.game_weather(
            stadium["lat"], stadium["lon"], game["game_time_utc"], stadium["azimuth_deg"]
        )
        for side, opp in (("home", "away"), ("away", "home")):
            team = game["teams"][side]
            opp_team = game["teams"][opp]
            pitcher_id = opp_team["probable_pitcher_id"]
            pitcher_hand = players.get(pitcher_id, {}).get("pitch_hand") if pitcher_id else None
            for slot, pid in enumerate(team["lineup"], start=1):
                info = players.get(pid, {})
                hand = _batter_hand(info.get("bat_side", "R"), pitcher_hand)
                stats = batters.loc[pid].to_dict() if pid in batters.index else {}
                split = _pitcher_split(splits, overall, pitcher_id, hand)
                pred = predict_player(stats, split, stadium, wx, hand, slot, league)
                rows.append({
                    "player_id": pid,
                    "game_pk": game["game_pk"],
                    "name": info.get("name", str(pid)),
                    "team": team["team_name"],
                    "opponent": opp_team["team_name"],
                    "venue": stadium["name"],
                    "game_time_utc": game["game_time_utc"],
                    "batter_hand": hand,
                    "pitcher_id": pitcher_id,
                    "pitcher_name": opp_team["probable_pitcher_name"],
                    "pitcher_hand": pitcher_hand,
                    "lineup_slot": slot,
                    "lineup_confirmed": team["lineup_confirmed"],
                    **pred,
                })

    rows.sort(key=lambda r: r["prob"], reverse=True)

    # attach odds to every scored player, not just the probability leaders —
    # the EV ranking can surface longshots priced too generously
    for r in rows:
        q = prop_odds.get(odds_mod.normalize_name(r["name"]))
        if q:
            r["odds"] = {
                **q,
                "implied_prob": round(q["implied_prob"], 4),
                "fair_prob": round(q["fair_prob"], 4),
                "ev_per_dollar": round(ev_per_dollar(r["prob"], q["best_price_decimal"]), 4),
                "edge_vs_fair": round(r["prob"] - q["fair_prob"], 4),
            }
        else:
            r["odds"] = None

    # ship the union of both top-N lists; the frontend re-sorts per view
    top_prob = rows[:top_n]
    top_ev = sorted(
        (r for r in rows if r["odds"]),
        key=lambda r: r["odds"]["ev_per_dollar"], reverse=True,
    )[:top_n]
    seen: set = set()
    top = []
    for r in top_prob + top_ev:
        k = (r["player_id"], r["game_time_utc"])
        if k not in seen:
            seen.add(k)
            top.append(r)
    top.sort(key=lambda r: r["prob"], reverse=True)

    doc = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds"),
        "date": date.isoformat(),
        "model_version": MODEL_VERSION,
        "league_hr_pa": round(league["hr_pa"], 4),
        "n_players_scored": len(rows),
        "odds_available": bool(prop_odds),
        "top_n": top_n,
        "players": top,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    safe_doc = _json_safe(doc)
    with open(output, "w") as f:
        # NaN (players missing from a stats table) is not valid JSON and
        # breaks the browser's fetch().json(); serialize it as null.
        json.dump(safe_doc, f, indent=2, allow_nan=False)
    # daily archive so pipeline/grade.py can score these picks tomorrow
    hist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "history")
    os.makedirs(hist_dir, exist_ok=True)
    with open(os.path.join(hist_dir, f"{date.isoformat()}.json"), "w") as f:
        json.dump(safe_doc, f, indent=2, allow_nan=False)
    log.info("wrote %s (%d players scored, top %d kept)", output, len(rows), len(top))
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
    ap.add_argument("--date", help="YYYY-MM-DD (default: today in US/Eastern)")
    ap.add_argument("--output", default="web/predictions.json")
    ap.add_argument("--lite", action="store_true", help="skip raw-statcast pitcher splits")
    ap.add_argument("--no-odds", action="store_true")
    ap.add_argument("--top", type=int, default=config.TOP_N)
    args = ap.parse_args()

    date = (
        dt.date.fromisoformat(args.date)
        if args.date
        else dt.datetime.now(ZoneInfo("America/New_York")).date()
    )
    run(date, args.output, args.lite, not args.no_odds, args.top)


if __name__ == "__main__":
    main()
