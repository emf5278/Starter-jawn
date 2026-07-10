"""Grade a past day's picks: how many of the top-20 (probability) and
top-20 (EV) actually went yard?

    python -m pipeline.grade [--date YYYY-MM-DD]   # default: yesterday (ET)

Reads the archived slate from history/<date>.json (written by pipeline.run),
pulls actual HR hitters from MLB StatsAPI box scores (Final games only), and
records:

  * results/log.csv        — one row per day, the running scoreboard
  * results/<date>.json    — pick-level detail (who hit, who didn't)

Columns in log.csv:
  date, snapshot_utc, prob_hits/prob_graded (top-20 by probability),
  prob_expected (sum of model probs — what the model *predicted* the hit
  count to be; calibration means hits ~ expected over time),
  ev_hits/ev_graded/ev_expected (top-20 by EV), ev_flat_pnl (P&L of $1 on
  every graded EV pick at the archived best price), late_snapshot (true if
  the slate was generated after ~2pm ET — odds/lineups may be mid-game).

Picks whose game never went Final (postponed/suspended) are excluded from
the graded counts. A player in a doubleheader is credited for an HR in
either game. The workflow runs this every morning before generating the
new slate; a missing archive (e.g. the very first day) just skips.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger("grade")

BASE = "https://statsapi.mlb.com/api/v1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR = os.path.join(ROOT, "history")
RESULTS_DIR = os.path.join(ROOT, "results")
LOG_CSV = os.path.join(RESULTS_DIR, "log.csv")

FIELDS = [
    "date", "snapshot_utc", "late_snapshot",
    "prob_hits", "prob_graded", "prob_expected",
    "ev_hits", "ev_graded", "ev_expected", "ev_flat_pnl",
]


def actual_hr_hitters(date: dt.date) -> tuple[dict[int, int], set[int]]:
    """(player_id -> HR that day, set of Final game_pks) from box scores."""
    sched = requests.get(
        f"{BASE}/schedule", params={"sportId": 1, "date": date.isoformat()}, timeout=30
    ).json()
    hrs: dict[int, int] = {}
    final_pks: set[int] = set()
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            pk = g["gamePk"]
            final_pks.add(pk)
            box = requests.get(f"{BASE}/game/{pk}/boxscore", timeout=30).json()
            for side in ("home", "away"):
                for pl in box["teams"][side]["players"].values():
                    hr = pl.get("stats", {}).get("batting", {}).get("homeRuns", 0)
                    if hr:
                        pid = pl["person"]["id"]
                        hrs[pid] = hrs.get(pid, 0) + hr
    return hrs, final_pks


def grade_picks(doc: dict, hrs: dict[int, int], final_pks: set[int]) -> dict:
    """Pure grading math — no network. Returns the log row + pick details."""
    n = doc.get("top_n", 20)
    players = doc["players"]
    top_prob = sorted(players, key=lambda p: p["prob"], reverse=True)[:n]
    top_ev = sorted(
        (p for p in players if p.get("odds")),
        key=lambda p: p["odds"]["ev_per_dollar"], reverse=True,
    )[:n]

    def detail(p: dict) -> dict:
        played = p.get("game_pk") in final_pks if p.get("game_pk") else True
        return {
            "name": p["name"], "team": p["team"], "prob": p["prob"],
            "played": played, "hr": hrs.get(p["player_id"], 0) if played else None,
            "odds": (p.get("odds") or {}).get("best_price_decimal"),
        }

    prob_rows = [detail(p) for p in top_prob]
    ev_rows = [detail(p) for p in top_ev]

    def hits(rows):  # graded = games that actually finished
        g = [r for r in rows if r["played"]]
        return sum(1 for r in g if r["hr"]), len(g), round(sum(r["prob"] for r in g), 2)

    prob_hits, prob_graded, prob_exp = hits(prob_rows)
    ev_hits, ev_graded, ev_exp = hits(ev_rows)
    pnl = sum(
        (r["odds"] - 1) if r["hr"] else -1.0
        for r in ev_rows if r["played"] and r["odds"]
    )

    gen = doc.get("generated_at", "")
    late = gen[11:13] >= "18" if len(gen) > 12 else False  # after ~2pm ET
    return {
        "row": {
            "date": doc["date"], "snapshot_utc": gen, "late_snapshot": late,
            "prob_hits": prob_hits, "prob_graded": prob_graded, "prob_expected": prob_exp,
            "ev_hits": ev_hits, "ev_graded": ev_graded, "ev_expected": ev_exp,
            "ev_flat_pnl": round(pnl, 2),
        },
        "top_prob": prob_rows,
        "top_ev": ev_rows,
    }


def append_log_row(row: dict) -> None:
    """Append to log.csv, replacing any existing row for the same date."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows: list[dict] = []
    if os.path.exists(LOG_CSV):
        with open(LOG_CSV, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["date"] != row["date"]]
    rows.append({k: str(row[k]) for k in FIELDS})
    rows.sort(key=lambda r: r["date"])
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="YYYY-MM-DD (default: yesterday in US/Eastern)")
    args = ap.parse_args()
    date = (
        dt.date.fromisoformat(args.date)
        if args.date
        else dt.datetime.now(ZoneInfo("America/New_York")).date() - dt.timedelta(days=1)
    )

    hist = os.path.join(HIST_DIR, f"{date.isoformat()}.json")
    if not os.path.exists(hist):
        log.info("no archived slate for %s (%s missing) — nothing to grade", date, hist)
        return
    with open(hist) as f:
        doc = json.load(f)

    log.info("grading %s picks against box scores", date)
    hrs, final_pks = actual_hr_hitters(date)
    if not final_pks:
        log.warning("no Final games found for %s; skipping", date)
        return

    res = grade_picks(doc, hrs, final_pks)
    append_log_row(res["row"])
    with open(os.path.join(RESULTS_DIR, f"{date.isoformat()}.json"), "w") as f:
        json.dump(res, f, indent=2)

    r = res["row"]
    log.info(
        "top-20 prob: %s/%s homered (model expected %s) | top-20 EV: %s/%s "
        "(expected %s), flat $1 P&L %+0.2f",
        r["prob_hits"], r["prob_graded"], r["prob_expected"],
        r["ev_hits"], r["ev_graded"], r["ev_expected"], r["ev_flat_pnl"],
    )


if __name__ == "__main__":
    main()
