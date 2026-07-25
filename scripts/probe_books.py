"""One-off diagnostic: which sportsbooks does our Odds API plan actually return?

    python scripts/probe_books.py

Prints, for each market we care about, the bookmakers returned in the `us` and
`us2` regions, plus the plan's quota headers. Answers three questions:

  1. Is the new (paid) key actually active?  -> x-requests-remaining
  2. Is FanDuel reachable at all, and in which region/market?
  3. Would adding `us2` to ODDS_REGIONS bring in books we're missing?

Read-only: touches no repo files and no pipeline state. Costs roughly 6
credits (the /events listing is free; each per-event or bulk odds request
costs markets x regions).
"""

from __future__ import annotations

import os
import sys

import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
TARGET = "fanduel"


def quota(r: requests.Response) -> str:
    h = r.headers
    return (f"[cost {h.get('x-requests-last','?')} | "
            f"used {h.get('x-requests-used','?')} | "
            f"remaining {h.get('x-requests-remaining','?')}]")


def show(label: str, r: requests.Response, books: list[str]) -> None:
    print(f"\n--- {label} -> HTTP {r.status_code} {quota(r)}")
    if r.status_code != 200:
        print(f"    ERROR BODY: {r.text[:400]}")
        return
    if not books:
        print("    (no bookmakers returned)")
        return
    print(f"    {len(books)} books: {', '.join(books)}")
    hit = [b for b in books if TARGET in b.lower()]
    print(f"    FanDuel present: {'YES -> ' + hit[0] if hit else 'no'}")


def books_of(payload) -> list[str]:
    """Bookmaker titles from either a single event dict or a list of events."""
    events = payload if isinstance(payload, list) else [payload]
    titles = set()
    for ev in events:
        for b in ev.get("bookmakers", []):
            titles.add(b.get("title") or b.get("key") or "?")
    return sorted(titles)


def main() -> None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("ODDS_API_KEY not set")

    ev = requests.get(f"{BASE}/sports/{SPORT}/events",
                      params={"apiKey": key}, timeout=30)
    print(f"events listing -> HTTP {ev.status_code} {quota(ev)}")
    ev.raise_for_status()
    events = ev.json()
    print(f"{len(events)} events on the board today")
    if not events:
        sys.exit("no events to probe")
    eid = events[0]["id"]
    print(f"probing event {eid}: {events[0].get('away_team')} @ {events[0].get('home_team')}")

    # Per-event player-prop markets (1 credit each: 1 market x 1 region)
    for market in ("batter_home_runs", "pitcher_strikeouts"):
        for region in ("us", "us2"):
            r = requests.get(
                f"{BASE}/sports/{SPORT}/events/{eid}/odds",
                params={"apiKey": key, "regions": region,
                        "markets": market, "oddsFormat": "decimal"},
                timeout=30)
            show(f"{market}  region={region}", r,
                 books_of(r.json()) if r.status_code == 200 else [])

    # Bulk main markets -- shows the widest book universe the plan can see
    for region in ("us", "us2"):
        r = requests.get(
            f"{BASE}/sports/{SPORT}/odds",
            params={"apiKey": key, "regions": region,
                    "markets": "h2h", "oddsFormat": "decimal"},
            timeout=30)
        show(f"h2h (all games)  region={region}", r,
             books_of(r.json()) if r.status_code == 200 else [])


if __name__ == "__main__":
    main()
