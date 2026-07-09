"""HR prop odds ("batter_home_runs", i.e. Over/Under 0.5 HR) from The Odds API.

De-vig math
-----------
A decimal price d implies probability 1/d, but the Over/Under pair sums to
more than 1 (the book's margin).  Per book we use the multiplicative method:

    fair_over = (1/d_over) / (1/d_over + 1/d_under)

and take the *median* fair_over across books as the market's consensus
probability.  If a book posts only the Over side (common for HR props), we
approximate fair_over = (1/d_over) / ASSUMED_SINGLE_SIDE_OVERROUND.

`best` is the highest Over price anywhere — the price you could actually bet.
"""

from __future__ import annotations

import logging
import statistics
import unicodedata

import requests

from .. import config

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4"


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalpha() or c == " ").strip()


def decimal_to_american(d: float) -> int:
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def fetch_hr_props(api_key: str, regions: str = "us") -> dict[str, dict]:
    """Map normalized player name -> odds summary for today's HR props."""
    try:
        events = requests.get(
            f"{BASE}/sports/{config.ODDS_SPORT_KEY}/events",
            params={"apiKey": api_key}, timeout=30,
        )
        events.raise_for_status()
        events = events.json()
    except Exception:
        log.warning("odds API events fetch failed", exc_info=True)
        return {}

    # per player: list of (book, over_price, under_price|None)
    quotes: dict[str, list[tuple[str, float, float | None]]] = {}
    for ev in events:
        try:
            r = requests.get(
                f"{BASE}/sports/{config.ODDS_SPORT_KEY}/events/{ev['id']}/odds",
                params={
                    "apiKey": api_key, "regions": regions,
                    "markets": config.ODDS_MARKET, "oddsFormat": "decimal",
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            log.warning("odds fetch failed for event %s", ev.get("id"), exc_info=True)
            continue
        for book in data.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != config.ODDS_MARKET:
                    continue
                # collect over/under per player for this book
                sides: dict[str, dict[str, float]] = {}
                for o in market.get("outcomes", []):
                    if o.get("point") not in (0.5, None):
                        continue
                    player = normalize_name(o.get("description", ""))
                    sides.setdefault(player, {})[o.get("name", "")] = o.get("price")
                for player, s in sides.items():
                    over = s.get("Over") or s.get("Yes")
                    if not over:
                        continue
                    quotes.setdefault(player, []).append(
                        (book.get("title", "?"), float(over), s.get("Under") or s.get("No"))
                    )

    out: dict[str, dict] = {}
    for player, qs in quotes.items():
        fair_probs = []
        for _, over, under in qs:
            imp_over = 1.0 / over
            if under:
                fair_probs.append(imp_over / (imp_over + 1.0 / float(under)))
            else:
                fair_probs.append(imp_over / config.ASSUMED_SINGLE_SIDE_OVERROUND)
        best_book, best_price, _ = max(qs, key=lambda q: q[1])
        out[player] = {
            "best_price_decimal": best_price,
            "best_price_american": decimal_to_american(best_price),
            "best_book": best_book,
            "implied_prob": 1.0 / best_price,
            "fair_prob": statistics.median(fair_probs),
            "n_books": len(qs),
        }
    return out
