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

Each factor slots in at its natural scope — some apply to every PA, some only
to the portion faced by the starter, some only to the bullpen portion:

| Factor | Scope | What it is | Code |
|---|---|---|---|
| `league_HR/PA` | — | League base rate (~0.032), computed live from the season's Statcast events | `data/statcast.py` |
| **B** — batter | every PA | Regressed barrels/PA, HR/FB, xISO vs league, weighted geometric mean `brl^0.45 · hrfb^0.30 · xiso^0.25` | `factors.batter_power_factor` |
| **K** — park | every PA | Handedness-specific HR park factor, `(PF/100)^0.85` | `factors.park_factor` |
| **W** — weather | every PA | `(1 + 0.008·(°F−70)) · (1 + 0.010·wind_out)`; wind projected onto the home-plate→CF bearing; domes neutral, retractable roofs half effect (or fully closed in extreme heat/cold) | `factors.weather_factor` |
| **T** — game total | every PA | Sportsbook total + run-line → implied team runs; `1 + 0.045·(runs − 4.5)`. Small on purpose (overlaps park/weather/pitching) | `factors.game_total_factor` |
| **P** — starter | starter PA | Starter's handedness-split HR/FB and FB-rate, shrunk hard (`^0.70`) | `factors.pitcher_hr_factor` |
| **BvP** — matchup | starter PA | This batter's career HR rate vs this starter, regressed **very** hard (ballast 100 PA) — see caveat below | `factors.bvp_factor` |
| **Pen** — bullpen | bullpen PA | Opposing bullpen's season HR/FB × a fatigue bump from its batters-faced over the last 3 games | `factors.bullpen_factor` |

Every raw rate is first **regressed to the mean** with a sample-size ballast —
`(obs·n + league·ballast)/(n + ballast)` — so a 40-PA hot streak can't print a
3× factor. Season-to-date is blended with last season (weighted 0.6) so April
isn't chaos.

> **A caveat on batter-vs-pitcher.** It's the most-requested and least-predictive
> input in this whole model. The sabermetric consensus (Tango/Lichtman/Dolphin's
> *The Book*, and every streakiness study since) is that BvP history carries
> almost no signal beyond the platoon split and overall quality already captured
> elsewhere — a 9-PA sample is a coin flip, not a trend. So it's deliberately
> throttled: a huge regression ballast, extra shrink (`^0.5`), and a tight
> `(0.85, 1.25)` cap. An 0-for-9 dings a hitter ~4%; it can never dominate a pick.
> It's in the model because it's real information and you asked for it — just
> weighted like the noisy signal it is.

Combining per-PA → per-game across the starter/bullpen split:

```
common    = league_HR/PA · B · K · W · T
p_starter = cap(common · P · BvP)      # ~62% of PA
p_bullpen = cap(common · Pen)          # ~38% of PA
P(≥1 HR)  = 1 − (1−p_starter)^(E[PA]·0.62) · (1−p_bullpen)^(E[PA]·0.38)
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
| pybaseball (Baseball Savant + FanGraphs) | barrels/PA, xISO, HR/FB, pitcher splits, bullpen HR-vuln, league rates | no |
| MLB StatsAPI | schedule, probable pitchers, confirmed lineups & batting order, handedness, batter-vs-pitcher history, bullpen usage | no |
| Open-Meteo | game-time temperature + wind at each stadium (lat/lon + field orientation in `pipeline/stadiums.py`) | no |
| The Odds API | HR prop odds **and** game totals/spreads | **yes** (props); totals/spreads also need it |

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

`.github/workflows/strikeouts.yml` (pitcher strikeout board) runs on its own
cron at **12:47 UTC (~8:47am ET)** — a few minutes offset from the HR
board's 8:41 run so the two don't queue at the same moment. Each pipeline
makes its own single Odds API call, so running both daily doesn't double up
on either board's usage. It projects games with first pitch at/after 5pm ET
and never touches `web/predictions.json`.

## Daily results logs

Every morning, before generating the new slate, each workflow grades
*yesterday's* picks against actual box scores. The two boards are graded
**completely separately** — different scripts, different CSVs — so a
strikeout losing streak never gets averaged into the HR numbers or vice versa.

**Home runs** (`pipeline/grade.py`):
* `results/log.csv` — one row per day: hits out of the top-20 by
  probability and top-20 by EV, the model's *expected* hit counts (if the
  model is calibrated, hits ≈ expected over time), and flat-$1 P&L on the
  EV list at the archived best prices.
* `results/<date>.json` — pick-level detail (who homered, who didn't).
* `history/<date>.json` — the archived slate each grade is based on.

**Pitcher strikeouts** (`pipeline/grade_strikeouts.py`):
* `results/strikeout_log.csv` — one row per day: hits out of the top-N by
  projected Ks and top-N by EV (graded on whichever side, Over or Under, the
  model favored), the model's expected hit counts, and flat-$1 P&L on the EV
  list at the archived best prices.
* `results/strikeouts/<date>.json` — pick-level detail (line, actual Ks,
  which side hit).
* `history/strikeouts/<date>.json` — the archived slate each grade is based on.

Rows flagged `late_snapshot=true` came from a slate generated after ~2pm ET
(a manual re-run) — odds and lineups may have been mid-game, so read those
rows skeptically. Picks whose game was postponed (HR) or pushed exactly on
the line (strikeouts) are excluded from grading.

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

Known simplifications: weather, game lines, batter-vs-pitcher, and bullpen
context are **not** reconstructed historically, so the backtest scores the core
park + batter + starter model (the newer factors default to neutral there).
League baselines use the full season. All documented in the script header.

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
