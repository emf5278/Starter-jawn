# MLB Home Run Predictor

A daily pipeline that estimates **P(player hits ≥1 HR today)** for every hitter
in a confirmed/projected lineup, compares those probabilities to de-vigged
sportsbook HR prop odds, and publishes the results to a static dashboard on
GitHub Pages — refreshed every morning by GitHub Actions. The dashboard
toggles between the **top 20 by model probability** and the **top 20 by EV**
at the best available price.

> Not betting advice. The model is deliberately simple and transparent so you
> can see exactly why it likes a player — and tune every weight yourself.

## The model

Everything is multiplicative around the league-average HR rate. For one plate
appearance:

```
p_PA = league_HR/PA × B × P × K × W
```

| Factor | What it is | Code |
|---|---|---|
| `league_HR/PA` | League base rate (~0.032), computed live from FanGraphs team totals | `pipeline/data/statcast.py` |
| **B** — batter | Regressed barrels/PA, HR/FB, xISO vs league, combined as a weighted geometric mean `brl^0.45 · hrfb^0.30 · xiso^0.25` | `factors.batter_power_factor` |
| **P** — pitcher | Starter's handedness-split HR/FB and FB-rate ratios (from raw Statcast), shrunk hard (`^0.70`) because pitchers control HR less than batters | `factors.pitcher_hr_factor` |
| **K** — park | Handedness-specific HR park factor, `(PF/100)^0.85` | `factors.park_factor` |
| **W** — weather | `(1 + 0.008·(°F−70)) · (1 + 0.010·wind_out_mph)`; wind is projected onto the home-plate→CF bearing; domes neutral, retractable roofs half effect | `factors.weather_factor` |

Every raw rate is first **regressed to the mean** with a sample-size ballast —
`(obs·n + league·ballast)/(n + ballast)` — so a 40-PA hot streak can't print a
3× factor. Season-to-date is blended with last season (weighted 0.6) so April
isn't chaos.

From per-PA to per-game, with the starter/bullpen split (the pitcher factor
only applies to the ~62% of PAs expected against the starter):

```
P(≥1 HR) = 1 − (1−p_starter)^(E[PA]·0.62) · (1−p_bullpen)^(E[PA]·0.38)
E[PA | lineup slot] = 4.65 − 0.115·(slot−1)
```

**Every constant above lives in [`pipeline/config.py`](pipeline/config.py)** —
ballasts, weights, elasticities, caps, weather coefficients, PA curve. Tune
there, re-run the backtest, compare Brier scores.

### Odds & EV

HR prop prices (`batter_home_runs`, i.e. Over 0.5 HR) come from
[The Odds API](https://the-odds-api.com). Per book we remove the vig
multiplicatively — `fair_over = (1/d_over) / (1/d_over + 1/d_under)` — and take
the median across books as the market's fair probability. Then, at the best
available price `d`:

```
EV per $1 = p_model·(d−1) − (1−p_model)
```

## Data sources

| Source | Used for | Key needed |
|---|---|---|
| pybaseball (Baseball Savant + FanGraphs) | barrels/PA, xISO, HR/FB, pitcher splits, league rates | no |
| MLB StatsAPI | schedule, probable pitchers, confirmed lineups & batting order, handedness | no |
| Open-Meteo | game-time temperature + wind at each stadium (lat/lon + field orientation in `pipeline/stadiums.py`) | no |
| The Odds API | HR prop odds | **yes** |

Stadium latitude/longitude, center-field compass bearing, roof type, and
handedness park factors are a hand-maintained table in
[`pipeline/stadiums.py`](pipeline/stadiums.py) (override park factors via
`data_cache/park_factors_override.csv`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your ODDS_API_KEY

# run today's predictions (full: pulls season-to-date raw Statcast, slow the
# first time, cached after)
python -m pipeline.run

# faster variant without pitcher handedness splits (FanGraphs overall instead)
python -m pipeline.run --lite --no-odds

# view the dashboard locally
python -m http.server -d web 8000    # → http://localhost:8000
```

Output goes to `web/predictions.json`: the union of the top 20 by probability
and the top 20 by EV, each with probability, best
odds, de-vigged implied probability, EV, and the full factor breakdown the
dashboard renders in each row's expandable "why" section. The checked-in file
is demo data (marked `"demo": true`) so the page renders before your first run.

## Automation (GitHub Actions + Pages)

`.github/workflows/daily.yml` runs at **13:00 UTC (~9am ET)** every day:
pipeline → commit `web/predictions.json` → deploy `web/` to GitHub Pages.

One-time repo setup:
1. **Settings → Pages → Source: GitHub Actions.**
2. **Settings → Secrets and variables → Actions**: add `ODDS_API_KEY`.
3. Trigger the first run from the Actions tab (`workflow_dispatch`; check
   "lite" for a quick first run). The pybaseball cache persists between runs
   via `actions/cache`, so the daily Statcast pull is incremental.

## Backtest

```bash
python backtest/backtest.py --season 2024
# with historical odds (The Odds API paid feature; bounded to control credits)
python backtest/backtest.py --season 2024 --odds --odds-days 14
```

Rebuilds each day's inputs cumulatively from raw Statcast events (batting
order and starters are inferred from the event stream; joins use
`merge_asof(allow_exact_matches=False)` so predictions never see same-day
data) and reports:

* **Calibration** — reliability curve (`backtest/output/calibration.png`),
  Brier score, log loss, base rate vs mean prediction
* **Top-10 hit rate** — how the dashboard's headline list would have done
* **ROI & CLV** (with `--odds`) — flat-stake ROI of +EV picks at the 13:00 UTC
  snapshot, and closing-line value vs the pre-game snapshot

Known simplifications: weather is not reconstructed historically, and league
baselines use the full season. Both are documented in the script header.

## Repo layout

```
pipeline/
  config.py          ← all tunable weights
  stadiums.py        ← park table (lat/lon, CF bearing, roof, HR factors)
  data/              ← statcast.py, lineups.py, weather.py, odds.py
  model/             ← factors.py (the math), predict.py (the combiner)
  run.py             ← daily entry point → web/predictions.json
web/                 ← index.html dashboard (vanilla JS + vendored Chart.js)
backtest/backtest.py ← season replay: calibration, ROI, CLV
.github/workflows/daily.yml
```
