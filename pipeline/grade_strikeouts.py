"""Grade a past day's pitcher-strikeout picks against actual box scores.

    python -m pipeline.grade_strikeouts [--date YYYY-MM-DD]   # default: yesterday (ET)

Reads the archived slate from history/strikeouts/<date>.json (written by
pipeline.run_strikeouts), pulls actual strikeout totals from MLB StatsAPI box
scores (Final games only), and records:

  * results/strikeout_log.csv   — one row per day, the running scoreboard
  * results/strikeouts/<date>.json — pick-level detail (line, actual Ks, hit/miss)

This is a completely separate log from the home-run board's results/log.csv —
the two boards are graded independently and never mixed.

Columns in strikeout_log.csv:
  date, snapshot_utc, late_snapshot,
  proj_hits/proj_graded/proj_expected (top-N by projected Ks, graded on
    whichever side — Over/Under — the model favored),
  ev_hits/ev_graded/ev_expected/ev_flat_pnl (top-N by best-side EV, same
    grading plus flat-$1 P&L at the archived best price).

A push (actual strikeouts exactly equal to an integer line) is excluded from
the graded counts, same as a postponed game. A missing archive (e.g. no
odds that day, or the board was never run) just skips.
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

log = logging.getLogger("grade_strikeouts")

BASE = "https://statsapi.mlb.com/api/v1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR = os.path.join(ROOT, "history", "strikeouts")
RESULTS_DIR = os.path.join(ROOT, "results")
DETAIL_DIR = os.path.join(RESULTS_DIR, "strikeouts")
LOG_CSV = os.path.join(RESULTS_DIR, "strikeout_log.csv")

FIELDS = [
    "date", "snapshot_utc", "late_snapshot",
    "proj_hits", "proj_graded", "proj_expected",
    "ev_hits", "ev_graded", "ev_expected", "ev_flat_pnl",
]


def actual_strikeouts(date: dt.date) -> tuple[dict[int, int], set[int]]:
    """(pitcher_id -> Ks that day, set of Final game_pks) from box scores."""
    sched = requests.get(
        f"{BASE}/schedule", params={"sportId": 1, "date": date.isoformat()}, timeout=30
    ).json()
    ks: dict[int, int] = {}
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
                    so = pl.get("stats", {}).get("pitching", {}).get("strikeOuts")
                    if so:
                        pid = pl["person"]["id"]
                        ks[pid] = ks.get(pid, 0) + so
    return ks, final_pks


def _actual_side(actual: int, line: float) -> str | None:
    if actual > line:
        return "Over"
    if actual < line:
        return "Under"
    return None  # push


def grade_picks(doc: dict, ks: dict[int, int], final_pks: set[int]) -> dict:
    """Pure grading math — no network. Returns the log row + pick details."""
    n = doc.get("top_n", 20)
    pitchers = doc["pitchers"]
    with_line = [p for p in pitchers if "line" in p]
    top_proj = sorted(with_line, key=lambda p: p["expected_ks"], reverse=True)[:n]
    top_ev = sorted(
        (p for p in with_line if p.get("odds")),
        key=lambda p: p["odds"]["best_ev"], reverse=True,
    )[:n]

    def detail(p: dict) -> dict:
        played = p.get("game_pk") in final_pks if p.get("game_pk") else True
        actual = ks.get(p["pitcher_id"]) if played else None
        side = "Over" if p["prob_over"] >= 0.5 else "Under"
        result = _actual_side(actual, p["line"]) if actual is not None else None
        return {
            "name": p["name"], "team": p["team"], "line": p["line"],
            "side": side, "played": played, "actual_ks": actual,
            "result": result, "hit": (result == side) if result else None,
            "price": (p.get("odds") or {}).get(
                "over_price_decimal" if side == "Over" else "under_price_decimal"),
            "model_prob": p["prob_over"] if side == "Over" else p["prob_under"],
        }

    proj_rows = [detail(p) for p in top_proj]
    ev_rows = [detail(p) for p in top_ev]

    def hits(rows):  # graded = games that finished and didn't push
        g = [r for r in rows if r["played"] and r["result"]]
        return sum(1 for r in g if r["hit"]), len(g), round(sum(r["model_prob"] for r in g), 2)

    proj_hits, proj_graded, proj_exp = hits(proj_rows)
    ev_hits, ev_graded, ev_exp = hits(ev_rows)
    pnl = sum(
        (r["price"] - 1) if r["hit"] else -1.0
        for r in ev_rows if r["played"] and r["result"] and r["price"]
    )

    gen = doc.get("generated_at", "")
    late = gen[11:13] >= "18" if len(gen) > 12 else False  # after ~2pm ET
    return {
        "row": {
            "date": doc["date"], "snapshot_utc": gen, "late_snapshot": late,
            "proj_hits": proj_hits, "proj_graded": proj_graded, "proj_expected": proj_exp,
            "ev_hits": ev_hits, "ev_graded": ev_graded, "ev_expected": ev_exp,
            "ev_flat_pnl": round(pnl, 2),
        },
        "top_proj": proj_rows,
        "top_ev": ev_rows,
    }


def append_log_row(row: dict) -> None:
    """Append to strikeout_log.csv, replacing any existing row for the same date."""
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
        log.info("no archived strikeout slate for %s (%s missing) — nothing to grade", date, hist)
        return
    with open(hist) as f:
        doc = json.load(f)

    if not doc.get("pitchers"):
        log.info("no pitchers in archived slate for %s — nothing to grade", date)
        return

    log.info("grading %s strikeout picks against box scores", date)
    ks, final_pks = actual_strikeouts(date)
    if not final_pks:
        log.warning("no Final games found for %s; skipping", date)
        return

    res = grade_picks(doc, ks, final_pks)
    append_log_row(res["row"])
    os.makedirs(DETAIL_DIR, exist_ok=True)
    with open(os.path.join(DETAIL_DIR, f"{date.isoformat()}.json"), "w") as f:
        json.dump(res, f, indent=2)

    r = res["row"]
    log.info(
        "top-%s projected: %s/%s hit (model expected %s) | top-%s EV: %s/%s "
        "(expected %s), flat $1 P&L %+0.2f",
        doc.get("top_n", 20), r["proj_hits"], r["proj_graded"], r["proj_expected"],
        doc.get("top_n", 20), r["ev_hits"], r["ev_graded"], r["ev_expected"], r["ev_flat_pnl"],
    )


if __name__ == "__main__":
    main()
