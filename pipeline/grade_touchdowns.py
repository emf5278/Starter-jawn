"""Grade a past day's anytime-TD picks against what actually happened.

    python -m pipeline.grade_touchdowns [--date YYYY-MM-DD]

Reads the archived slate from history/touchdowns/<date>.json (written by
pipeline.run_touchdowns), pulls the day's scorers from nflverse play-by-play,
and records:

  * results/touchdown_log.csv     — one row per slate, the running scoreboard
  * results/touchdowns/<date>[-<window>].json — pick-level detail

Sunday publishes twice (an early board and a late one) and each is archived
and graded separately, so the log keys on (date, window) and a Sunday
contributes two rows rather than one overwriting the other.

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

FIELDS = ["date", "window", "season", "week", "snapshot_utc", "cold_start",
          "prob_hits", "prob_graded", "prob_expected",
          "ev_hits", "ev_graded", "ev_expected", "ev_flat_pnl"]


def _archives_for(date: dt.date) -> list[str]:
    """Every archive for a date.

    Sunday is published twice -- an early board and a late one -- and each is
    archived separately, so a date can have more than one slate to grade.
    """
    import glob
    stem = date.isoformat()
    paths = sorted(set(
        glob.glob(os.path.join(HIST_DIR, f"{stem}.json"))
        + glob.glob(os.path.join(HIST_DIR, f"{stem}-*.json"))))
    if not paths:
        log.info("no archived slate for %s", date)
    return paths


def _load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        log.warning("could not read %s", path, exc_info=True)
        return None


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


def run(date: dt.date) -> list[dict]:
    paths = _archives_for(date)
    if not paths:
        return []
    scorers = nfl.results_for(date)
    if not scorers or not any(scorers.values()):
        log.info("no play-by-play for %s yet — try again tomorrow", date)
        return []
    rows = []
    for path in paths:
        row = _grade_one(date, path, scorers)
        if row:
            rows.append(row)
    return rows


def _grade_one(date: dt.date, path: str, scorers: dict) -> dict | None:
    doc = _load(path)
    if not doc or doc.get("skipped"):
        return None
    players = doc.get("players", [])
    if not players:
        log.info("archived slate %s has no players", os.path.basename(path))
        return None
    window = doc.get("window", "all")

    top_n = doc.get("top_n", 20)
    floor = doc.get("ev_min_prob", 0) or 0
    by_prob = sorted(players, key=lambda p: p["prob"], reverse=True)[:top_n]
    by_ev = sorted((p for p in players if p.get("odds") and p["prob"] >= floor),
                   key=lambda p: p["odds"]["ev_per_dollar"], reverse=True)[:top_n]

    p_hits, p_graded, p_exp, _ = _grade_list(by_prob, scorers)
    e_hits, e_graded, e_exp, e_pnl = _grade_list(by_ev, scorers)

    row = {
        "date": date.isoformat(),
        "window": window,
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
    stem = date.isoformat() if window == "all" else f"{date.isoformat()}-{window}"
    with open(os.path.join(DETAIL_DIR, f"{stem}.json"), "w") as f:
        json.dump({"date": date.isoformat(), "window": window,
                   "season": doc.get("season"),
                   "week": doc.get("week"), "picks": detail}, f, indent=2)

    log.info("%s [%s]: top-%d by probability %d/%d (expected %.1f) | "
             "by EV %d/%d (expected %.1f, flat P&L %+.2f)",
             date, window, top_n, p_hits, p_graded, p_exp,
             e_hits, e_graded, e_exp, e_pnl)
    return row


def _append(row: dict) -> None:
    """Append to touchdown_log.csv, replacing any row for the same slate.

    Keyed on (date, window), not date alone: a Sunday produces an early row
    and a late row, and neither should overwrite the other.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    key = (row["date"], row.get("window", "all"))
    rows = []
    if os.path.exists(LOG_CSV):
        with open(LOG_CSV) as f:
            rows = [r for r in csv.DictReader(f)
                    if (r.get("date"), r.get("window", "all")) != key]
    rows.append({k: row.get(k) for k in FIELDS})
    rows.sort(key=lambda r: (r.get("date") or "", r.get("window") or ""))
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
