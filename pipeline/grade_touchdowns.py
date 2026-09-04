"""Grade a past day's anytime-TD picks against what actually happened.

    python -m pipeline.grade_touchdowns [--date YYYY-MM-DD]

Reads the archived slate from history/touchdowns/<date>.json (written by
pipeline.run_touchdowns), pulls the day's scorers from nflverse play-by-play,
and records:

  * results/touchdown_log.csv     — one row per slate, the running scoreboard
  * results/touchdowns/<date>.json — pick-level detail (scored / didn't)

Completely separate from the HR and strikeout logs — different sport,
different file, never averaged together.

Columns in touchdown_log.csv:
  date, season, week, snapshot_utc, cold_start,
  prob_hits/prob_graded/prob_expected  (top-N by model probability),
  ev_hits/ev_graded/ev_expected/ev_flat_pnl  (top-N by EV, plus flat-$1 P&L
    at the archived best price)

If the model is calibrated, hits should track expected over time — that is
the whole point of logging expected alongside actual.

Play-by-play for a given week is published a day or two after the games, so
grading a Sunday slate on Monday morning can legitimately find nothing yet;
the run exits cleanly and the next day's run picks it up.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
from zoneinfo import ZoneInfo

from .data import nfl

log = logging.getLogger("grade_touchdowns")

ET = ZoneInfo("America/New_York")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR = os.path.join(ROOT, "history", "touchdowns")
RESULTS_DIR = os.path.join(ROOT, "results")
DETAIL_DIR = os.path.join(RESULTS_DIR, "touchdowns")
LOG_CSV = os.path.join(RESULTS_DIR, "touchdown_log.csv")

FIELDS = ["date", "season", "week", "snapshot_utc", "cold_start",
          "prob_hits", "prob_graded", "prob_expected",
          "ev_hits", "ev_graded", "ev_expected", "ev_flat_pnl"]


def _load_slate(date: dt.date) -> dict | None:
    path = os.path.join(HIST_DIR, f"{date.isoformat()}.json")
    if not os.path.exists(path):
        log.info("no archived slate for %s", date)
        return None
    with open(path) as f:
        return json.load(f)


def _grade_list(picks: list[dict], scorers: dict[str, set]) -> tuple[int, int, float, float]:
    """-> (hits, graded, expected_hits, flat_pnl)"""
    hits = graded = 0
    expected = pnl = 0.0
    for p in picks:
        gid = p.get("game_id")
        if gid not in scorers:
            continue                      # game not in the data yet
        graded += 1
        scored = p["player_id"] in scorers[gid]
        hits += int(scored)
        expected += p["prob"]
        o = p.get("odds")
        if o and o.get("best_price_decimal"):
            pnl += (o["best_price_decimal"] - 1.0) if scored else -1.0
    return hits, graded, expected, pnl


def run(date: dt.date) -> dict | None:
    doc = _load_slate(date)
    if not doc or doc.get("skipped"):
        return None
    players = doc.get("players", [])
    if not players:
        log.info("archived slate for %s has no players", date)
        return None

    scorers = nfl.results_for(date)
    if not scorers or not any(scorers.values()):
        log.info("no play-by-play for %s yet — try again tomorrow", date)
        return None

    top_n = doc.get("top_n", 20)
    by_prob = sorted(players, key=lambda p: p["prob"], reverse=True)[:top_n]
    by_ev = sorted((p for p in players if p.get("odds")),
                   key=lambda p: p["odds"]["ev_per_dollar"], reverse=True)[:top_n]

    p_hits, p_graded, p_exp, _ = _grade_list(by_prob, scorers)
    e_hits, e_graded, e_exp, e_pnl = _grade_list(by_ev, scorers)

    row = {
        "date": date.isoformat(),
        "season": doc.get("season"),
        "week": doc.get("week"),
        "snapshot_utc": doc.get("generated_at"),
        "cold_start": doc.get("cold_start"),
        "prob_hits": p_hits, "prob_graded": p_graded, "prob_expected": round(p_exp, 2),
        "ev_hits": e_hits, "ev_graded": e_graded, "ev_expected": round(e_exp, 2),
        "ev_flat_pnl": round(e_pnl, 2),
    }
    _append(row)

    detail = []
    for p in players:
        gid = p.get("game_id")
        if gid not in scorers:
            continue
        detail.append({
            "player_id": p["player_id"], "name": p["name"],
            "position": p.get("position"), "team": p.get("team"),
            "opponent": p.get("opponent"), "prob": p["prob"],
            "confidence": p.get("confidence"),
            "odds": p.get("odds"),
            "scored": p["player_id"] in scorers[gid],
        })
    os.makedirs(DETAIL_DIR, exist_ok=True)
    with open(os.path.join(DETAIL_DIR, f"{date.isoformat()}.json"), "w") as f:
        json.dump({"date": date.isoformat(), "season": doc.get("season"),
                   "week": doc.get("week"), "picks": detail}, f, indent=2)

    log.info("%s: top-%d by probability %d/%d (expected %.1f) | "
             "by EV %d/%d (expected %.1f, flat P&L %+.2f)",
             date, top_n, p_hits, p_graded, p_exp, e_hits, e_graded, e_exp, e_pnl)
    return row


def _append(row: dict) -> None:
    """Append to touchdown_log.csv, replacing any existing row for that date."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    if os.path.exists(LOG_CSV):
        with open(LOG_CSV) as f:
            rows = [r for r in csv.DictReader(f) if r.get("date") != row["date"]]
    rows.append({k: row.get(k) for k in FIELDS})
    rows.sort(key=lambda r: r.get("date") or "")
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="YYYY-MM-DD (default: yesterday, ET)")
    args = ap.parse_args()
    date = (dt.date.fromisoformat(args.date) if args.date
            else dt.datetime.now(ET).date() - dt.timedelta(days=1))
    run(date)


if __name__ == "__main__":
    main()
