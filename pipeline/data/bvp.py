"""Career batter-vs-starter history from MLB StatsAPI (public, no key).

The `vsPlayerTotal` hitting split gives a batter's aggregate line against one
specific pitcher across all seasons — the "Harper is 0-for-5 with 9 PA vs
Flaherty" number.  One HTTP call per (batter, pitcher) pair, so we thread the
slate's ~250-300 matchups.  Everything is best-effort: any failure yields no
entry, and the model treats a missing matchup as neutral.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import requests

log = logging.getLogger(__name__)

BASE = "https://statsapi.mlb.com/api/v1"
_MAX_WORKERS = 8


def _fetch_one(session: requests.Session, batter_id: int, pitcher_id: int) -> dict:
    """{"pa", "hr", "ab"} for this batter vs this pitcher (career), or zeros."""
    try:
        r = session.get(
            f"{BASE}/people/{batter_id}/stats",
            params={"stats": "vsPlayerTotal", "group": "hitting",
                    "opposingPlayerId": pitcher_id, "sportId": 1},
            timeout=15,
        )
        r.raise_for_status()
        for grp in r.json().get("stats", []):
            for sp in grp.get("splits", []):
                st = sp.get("stat", {})
                pa = st.get("plateAppearances")
                if pa:
                    return {"pa": int(pa), "hr": int(st.get("homeRuns", 0)),
                            "ab": int(st.get("atBats", 0))}
    except Exception:
        log.debug("bvp lookup failed for %s vs %s", batter_id, pitcher_id, exc_info=True)
    return {"pa": 0, "hr": 0, "ab": 0}


def fetch_bvp(pairs: list[tuple[int, int]]) -> dict[tuple[int, int], dict]:
    """Map (batter_id, pitcher_id) -> {"pa","hr","ab"} for each unique pair."""
    uniq = sorted({p for p in pairs if p[0] and p[1]})
    if not uniq:
        return {}
    out: dict[tuple[int, int], dict] = {}
    with requests.Session() as session:
        def work(pair):
            return pair, _fetch_one(session, pair[0], pair[1])
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            for pair, res in ex.map(work, uniq):
                out[pair] = res
    n_hist = sum(1 for v in out.values() if v["pa"] > 0)
    log.info("bvp: %d/%d matchups have prior history", n_hist, len(out))
    return out
