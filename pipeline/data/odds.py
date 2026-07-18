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


def fetch_game_lines(api_key: str, regions: str = "us") -> dict[str, dict]:
    """Map normalized team name -> {total, spread, implied_team_runs}.

    One call to the main-markets endpoint (totals + spreads) — far cheaper and
    better-covered than player props.  Per game we take the median total and
    each team's median spread across books, then

        implied_team_runs = total/2 - team_spread/2

    (a favored team's run-line spread is negative, so it implies more runs).
    """
    try:
        r = requests.get(
            f"{BASE}/sports/{config.ODDS_SPORT_KEY}/odds",
            params={"apiKey": api_key, "regions": regions,
                    "markets": config.ODDS_GAME_MARKETS, "oddsFormat": "decimal"},
            timeout=30,
        )
        r.raise_for_status()
        games = r.json()
    except Exception:
        log.warning("game-line fetch failed", exc_info=True)
        return {}

    out: dict[str, dict] = {}
    for g in games:
        totals: list[float] = []
        spreads: dict[str, list[float]] = {}
        for book in g.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] == "totals":
                    pts = [o.get("point") for o in market.get("outcomes", [])
                           if o.get("point") is not None]
                    if pts:
                        totals.append(float(pts[0]))
                elif market["key"] == "spreads":
                    for o in market.get("outcomes", []):
                        if o.get("point") is not None:
                            spreads.setdefault(o["name"], []).append(float(o["point"]))
        if not totals:
            continue
        total = statistics.median(totals)
        for team in (g.get("home_team"), g.get("away_team")):
            sp = statistics.median(spreads[team]) if spreads.get(team) else None
            itr = (total / 2 - sp / 2) if sp is not None else None
            out[normalize_name(team)] = {
                "total": round(total, 1),
                "spread": round(sp, 1) if sp is not None else None,
                "implied_team_runs": round(itr, 2) if itr is not None else None,
            }
    log.info("game lines fetched for %d teams", len(out))
    return out


def fetch_strikeout_props(api_key: str, regions: str = "us") -> dict[str, dict]:
    """Map normalized pitcher name -> strikeout Over/Under summary.

    Unlike HR props (a fixed 0.5 line), strikeout props carry a real total
    (e.g. 6.5) that can differ by book.  Per pitcher we pick the **modal line**
    (the total the most books agree on), then within that line take the best
    Over price and best Under price available, and the median de-vigged
    P(Over) across books:

        fair_over = (1/d_over) / (1/d_over + 1/d_under)
    """
    try:
        events = requests.get(
            f"{BASE}/sports/{config.ODDS_SPORT_KEY}/events",
            params={"apiKey": api_key}, timeout=30,
        )
        events.raise_for_status()
        events = events.json()
    except Exception:
        log.warning("odds API events fetch failed (strikeouts)", exc_info=True)
        return {}

    # per pitcher -> per line -> list of (over_price, under_price)
    quotes: dict[str, dict[float, list[tuple[float, float | None]]]] = {}
    for ev in events:
        try:
            r = requests.get(
                f"{BASE}/sports/{config.ODDS_SPORT_KEY}/events/{ev['id']}/odds",
                params={"apiKey": api_key, "regions": regions,
                        "markets": config.ODDS_K_MARKET, "oddsFormat": "decimal"},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            log.warning("strikeout odds fetch failed for event %s", ev.get("id"), exc_info=True)
            continue
        for book in data.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != config.ODDS_K_MARKET:
                    continue
                sides: dict[tuple[str, float], dict[str, float]] = {}
                for o in market.get("outcomes", []):
                    pt = o.get("point")
                    if pt is None:
                        continue
                    key = (normalize_name(o.get("description", "")), float(pt))
                    sides.setdefault(key, {})[o.get("name", "")] = o.get("price")
                for (player, pt), s in sides.items():
                    over, under = s.get("Over"), s.get("Under")
                    if not over:
                        continue
                    quotes.setdefault(player, {}).setdefault(pt, []).append(
                        (float(over), float(under) if under else None))

    out: dict[str, dict] = {}
    for player, by_line in quotes.items():
        # modal line = the total quoted by the most books
        line = max(by_line, key=lambda k: len(by_line[k]))
        rows = by_line[line]
        best_over = max(o for o, _ in rows)
        unders = [u for _, u in rows if u]
        best_under = max(unders) if unders else None
        fair = []
        for over, under in rows:
            io = 1.0 / over
            fair.append(io / (io + 1.0 / under) if under
                        else io / config.ASSUMED_SINGLE_SIDE_OVERROUND)
        out[player] = {
            "line": line,
            "over_price_decimal": best_over,
            "over_price_american": decimal_to_american(best_over),
            "under_price_decimal": best_under,
            "under_price_american": decimal_to_american(best_under) if best_under else None,
            "fair_over": statistics.median(fair),
            "n_books": len(rows),
        }
    return out
