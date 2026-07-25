"""Pitcher-strikeout board entry point (separate from the HR pipeline).

    python -m pipeline.run_strikeouts [--date YYYY-MM-DD]
                                      [--output web/strikeouts.json] [--no-odds]
                                      [--all-day]

Projects each probable starter's strikeout total for tonight's games (first
pitch at/after 5pm ET by default), computes P(Over) vs the sportsbook line, and
writes a ranked JSON the strikeouts.html page renders. Reuses the same data
layer (StatsAPI slate, cached Statcast events, The Odds API) as the HR board;
it does NOT touch web/predictions.json.
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
from .model import strikeouts as K
from .model.predict import ev_per_dollar

log = logging.getLogger("pipeline.k")

MODEL_VERSION = "k-0.1.0"
ET = ZoneInfo("America/New_York")


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _after_cutoff(game_time_utc: str, all_day: bool) -> bool:
    if all_day or not game_time_utc:
        return True
    try:
        et = dt.datetime.fromisoformat(game_time_utc.replace("Z", "+00:00")).astimezone(ET)
    except Exception:
        return True
    return et.hour >= config.K_MIN_START_ET_HOUR


def run(date: dt.date, output: str, use_odds: bool, all_day: bool) -> dict:
    season = date.year
    log.info("fetching slate + lineups for %s", date)
    slate = lineups.todays_slate(date)
    players = slate["players"]

    log.info("fetching raw statcast events (cached)")
    events = statcast.season_events(season, date)
    if events is None:
        raise SystemExit("no Statcast events available; cannot project strikeouts")
    league = {"k_pa": statcast.league_k_rate_from_events(events)}
    pk_stats = statcast.pitcher_k_stats_from_events(events)
    bk_stats = statcast.batter_k_from_events(events)
    log.info("league K/PA=%.3f | %d pitchers, %d batters with K data",
             league["k_pa"], len(pk_stats), len(bk_stats))

    k_props = {}
    if use_odds and os.environ.get("ODDS_API_KEY"):
        log.info("fetching strikeout prop odds")
        k_props = odds_mod.fetch_strikeout_props(
            os.environ["ODDS_API_KEY"], os.environ.get("ODDS_REGIONS", "us"))
        log.info("strikeout props for %d pitchers", len(k_props))
    elif use_odds:
        log.warning("ODDS_API_KEY not set; skipping strikeout odds")

    def opp_batters(lineup: list[int]) -> list[tuple]:
        out = []
        for slot, bid in enumerate(lineup, start=1):
            w = config.expected_pa_for_slot(slot)
            if bid in bk_stats.index:
                row = bk_stats.loc[bid]
                out.append((float(row["k_pa"]), float(row["pa"]), w))
            else:
                out.append((None, 0, w))  # unknown hitter -> regresses to league
        return out

    rows = []
    for game in slate["games"]:
        if not _after_cutoff(game.get("game_time_utc"), all_day):
            continue
        for side, opp in (("home", "away"), ("away", "home")):
            team, opp_team = game["teams"][side], game["teams"][opp]
            pid = team["probable_pitcher_id"]
            if not pid:
                continue
            pkrow = pk_stats.loc[pid] if pid in pk_stats.index else None
            pk = K.pitcher_k_factor(
                float(pkrow["k_pa"]) if pkrow is not None else None,
                float(pkrow["pa"]) if pkrow is not None else None, league)
            tbf = K.expected_tbf(
                float(pkrow["tbf_per_start"]) if pkrow is not None and not math.isnan(pkrow.get("tbf_per_start", float("nan"))) else None,
                float(pkrow["starts"]) if pkrow is not None else None)
            opp_f = K.opponent_k_factor(opp_batters(opp_team["lineup"]), league)

            q = k_props.get(odds_mod.normalize_name(team["probable_pitcher_name"] or ""))
            line = q["line"] if q else None
            pred = K.predict_pitcher(pk, opp_f, tbf, line, league)

            row = {
                "pitcher_id": pid,
                "name": team["probable_pitcher_name"] or str(pid),
                "team": team["team_name"],
                "opponent": opp_team["team_name"],
                "opponent_lineup_confirmed": opp_team["lineup_confirmed"],
                "game_pk": game["game_pk"],
                "game_time_utc": game["game_time_utc"],
                "pitcher_hand": players.get(pid, {}).get("pitch_hand"),
                **pred,
            }
            # Break-even price per side: the shortest odds at which that side
            # is still +EV on the model's numbers. Book-independent, so it's
            # usable at a sportsbook the odds feed doesn't carry.
            for side in ("over", "under"):
                p = pred.get(f"prob_{side}")
                if p:
                    row[f"break_even_{side}_american"] = \
                        odds_mod.decimal_to_american(1.0 / p)
                else:
                    row[f"break_even_{side}_american"] = None
            if q:
                p_over, p_under = pred["prob_over"], pred["prob_under"]
                ev_over = ev_per_dollar(p_over, q["over_price_decimal"])
                ev_under = (ev_per_dollar(p_under, q["under_price_decimal"])
                            if q["under_price_decimal"] else None)
                best_side, best_ev = ("Over", ev_over)
                if ev_under is not None and ev_under > ev_over:
                    best_side, best_ev = ("Under", ev_under)
                row["odds"] = {
                    **q,
                    "fair_over": round(q["fair_over"], 4),
                    "ev_over": round(ev_over, 4),
                    "ev_under": round(ev_under, 4) if ev_under is not None else None,
                    "best_side": best_side,
                    "best_ev": round(best_ev, 4),
                    "edge_over": round(p_over - q["fair_over"], 4),
                }
            else:
                row["odds"] = None
            rows.append(row)

    rows.sort(key=lambda r: r["expected_ks"], reverse=True)
    top_proj = rows[:config.K_TOP_N]
    top_ev = sorted((r for r in rows if r["odds"]),
                    key=lambda r: r["odds"]["best_ev"], reverse=True)[:config.K_TOP_N]
    seen, top = set(), []
    for r in top_proj + top_ev:
        if r["pitcher_id"] not in seen:
            seen.add(r["pitcher_id"])
            top.append(r)
    top.sort(key=lambda r: r["expected_ks"], reverse=True)

    doc = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds"),
        "date": date.isoformat(),
        "model_version": MODEL_VERSION,
        "league_k_pa": round(league["k_pa"], 4),
        "filter": "first pitch ≥ 5pm ET" if not all_day else "all games",
        "n_starters_scored": len(rows),
        "odds_available": bool(k_props),
        "top_n": config.K_TOP_N,
        "pitchers": top,
    }
    safe = _json_safe(doc)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(safe, f, indent=2, allow_nan=False)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hist = os.path.join(root, "history", "strikeouts")
    os.makedirs(hist, exist_ok=True)
    with open(os.path.join(hist, f"{date.isoformat()}.json"), "w") as f:
        json.dump(safe, f, indent=2, allow_nan=False)
    log.info("wrote %s (%d starters scored, top %d kept)", output, len(rows), len(top))
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
    ap.add_argument("--output", default="web/strikeouts.json")
    ap.add_argument("--no-odds", action="store_true")
    ap.add_argument("--all-day", action="store_true", help="include games before 5pm ET too")
    args = ap.parse_args()
    date = (dt.date.fromisoformat(args.date) if args.date
            else dt.datetime.now(ET).date())
    run(date, args.output, not args.no_odds, args.all_day)


if __name__ == "__main__":
    main()
