# MLB & NFL Prop Model

Transparent prop models that estimate a player's probability of doing a
thing today, compare it to de-vigged sportsbook prices, and publish the
result to a static dashboard on GitHub Pages — refreshed by GitHub Actions.
Every board toggles between the **top 20 by model probability** and the
**top 20 by EV** at the best available price.

| Board | Question | Page |
|---|---|---|
| Home runs | P(hitter hits ≥1 HR today) | `index.html` |
| Pitcher strikeouts | P(starter goes Over his K line) | `strikeouts.html` |
| Moneylines & totals | P(home win), P(Over) | `moneylines.html`, `totals.html` |
| **NFL anytime TD** | **P(player scores ≥1 TD today)** | **`touchdowns.html`** |

Most of this README describes the home-run model, which came first and sets
the pattern the others follow; the NFL touchdown board has [its own
section](#nfl-anytime-touchdown-scorers).

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

`.github/workflows/daily.yml, touchdowns.yml` runs the HR pipeline → commits
`web/predictions.json` → deploys `web/` to GitHub Pages.
`.github/workflows/strikeouts.yml` does the same for the pitcher-strikeout
board (`web/strikeouts.json`), on its own cron and its own single Odds API
call — running both daily doesn't double up on either board's usage. It
projects games with first pitch at/after 5pm ET and never touches
`web/predictions.json`.

**Self-healing schedule.** GitHub's `schedule` trigger for this repo has
proven unreliable on its own — some mornings it fires hours late, some
mornings it silently never fires. Rather than one cron per board and a
human noticing when it fails, each workflow now lists **five cron entries**
spread across the morning (roughly every 30 minutes, offset a few minutes
from each other so the two boards don't queue at the same moment). A
"Decide whether to run" gate step at the top of each job checks whether
`web/predictions.json` / `web/strikeouts.json` already shows today's date
(US/Eastern); if so, it skips the entire rest of the job — no Statcast
pull, no Odds API call, no commit — and exits in a few seconds. Only the
first attempt that actually fires that day does real work, so having
multiple scheduled entries costs nothing on a normal day and guarantees
the board still refreshes on a day GitHub drops most of them. Manual
`workflow_dispatch` always runs regardless of what's already committed
(so a late-day lineup/weather re-check still works as before).

One-time repo setup:
1. **Settings → Pages → Source: GitHub Actions.**
2. **Settings → Secrets and variables → Actions**: add `ODDS_API_KEY`.
3. Trigger the first run from the Actions tab (`workflow_dispatch`; check
   "lite" for a quick first run). The pybaseball cache persists between runs
   via `actions/cache`, so the daily Statcast pull is incremental.

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

## NFL anytime touchdown scorers

A fourth board, same concept as the home-run one but for
`player_anytime_td`: **P(player scores >=1 rushing or receiving TD today)**,
ranked by model probability and by EV at the best available price.
Published to `web/touchdowns.json` and rendered by `web/touchdowns.html`.

### The model

Anchored on the game line and divided by usage:

```
implied_points = total/2 - spread/2                    # same math as the MLB game lines
team_off_TD    = 0.12479 * implied_points - 0.4434     # OLS on 2025 team-games, r = 0.879
team_rush_TD   = team_off_TD * team_rush_share         # team's own run/pass TD split, regressed
lambda         = team_rush_TD * rush_share * opp_rush
               + team_rec_TD  * rec_share  * opp_rec
P(anytime TD)  = 1 - exp(-lambda)
```

Poisson rather than binomial because a player can score twice; ~13% of
scoring games are multi-TD games, so treating it as a coin flip would bias
every probability upward.

**The share terms are where the work is.** A player's cut of his team's
touchdowns is driven by *volume*, not by last year's scoring. Walk-forward
tested on 2024 + 2025 (weeks 8-18, predicting each week from prior weeks
only), single-signal AUCs:

| | inside-5 carry share | carry share | prior TD share |
|---|---|---|---|
| RB/FB | .681 | **.685** | .666 |
| QB | **.638** | .630 | .630 |
| WR/TE (targets / RZ targets) | .666 | **.672** | .661 |

Carries from inside the 5 are 5.6% of all carries but produce 59% of rushing
touchdowns; targets inside the 20 are 14% of targets and produce 71% of
receiving touchdowns. So the model weights goal-line and red-zone usage
heavily and prior touchdowns barely at all — a grid search put prior rushing
TD share's weight at **zero** for backs, which is the standard TD-regression
result.

**Quarterbacks get their own weights.** Passing touchdowns never count
toward an anytime-TD prop, so QBs ride on rushing alone — and for them the
signal inverts: carry share is worthless (it is mostly scrambles and
kneel-downs) while TD share carries real information about whether a team
hands him the ball on the one. Pooling the positions, which one weight
vector does, systematically underrates goal-line quarterbacks: the eight
QBs with the highest goal-line carry share scored in **37.5%** of their
weeks against an 18.0% base rate.

Shares are shrunk toward a replacement-level body by opportunity count, then
**renormalised across the players actually on the roster** — which is what
redistributes a departed starter's work to whoever inherits it, and what
makes a mid-season trade show up immediately. Rookies have no usage at all,
so draft slot is their prior.

### Calibration

`python -m pipeline.calibrate_td --season 2025` replays the season week by
week, rebuilding every input from prior weeks only. On 2025 weeks 8-18
(4,389 predictions, 14.9% base rate):

| | model | base rate | skill |
|---|---|---|---|
| Brier | 0.11244 | 0.12649 | **+11.1%** |
| log loss | 0.36736 | 0.42019 | **+12.6%** |

Reliability holds across most of the range — predicted vs observed is within
0.02 in seven of nine probability bins. The two constants that are *not*
guesses but fitted here are the replacement-level share and the share
concentration exponent; the comments in `config.py` record what each one was
worth. Top-20-by-probability hit 43.2% against a model expectation of 47.9%,
so the very top of the board still runs slightly hot — partly real, partly
because this harness holds team touchdowns at the league average instead of
using a game line, and has no inactives data.

### Honest limitations

* **No injury or inactives feed.** nflverse has not published 2026 injuries
  yet, so a player ruled out 90 minutes before kickoff still appears on the
  board. This is the single biggest gap.
* **Week 1 is a cold start.** Before any current-season snaps exist the
  model runs entirely on last season, which cannot see free agency, the
  draft or a camp battle. Every row is badged `low` confidence until the
  season has enough games, and the page carries a banner saying so.
* **Props coverage is thin.** Anytime-TD prices come from whatever books
  the Odds API plan carries; where only the "Yes" side is quoted the "market
  fair" column assumes a 6% overround rather than measuring one, so it is an
  estimate, not a true no-vig line.

### Running it

```bash
python -m pipeline.run_touchdowns                 # today's slate -> web/touchdowns.json
python -m pipeline.run_touchdowns --date 2026-09-13 --no-odds
python -m pipeline.grade_touchdowns --date 2026-09-13
python -m pipeline.calibrate_td --season 2025     # walk-forward calibration
```

Data comes from [nflverse](https://github.com/nflverse) — schedule, weekly
rosters and play-by-play, all free and keyless — cached under
`data_cache/nfl`. Only the odds need `ODDS_API_KEY`.

### Automation and API credits

`.github/workflows/touchdowns.yml` runs on the same self-healing multi-cron
pattern as the other boards, but an NFL slate is a calendar day, so most
days there is nothing to do. A gate step checks the published schedule
**before** setting up Python or touching The Odds API and exits in seconds
on a Tuesday; player props are then requested only for games kicking off
that day. That works out to roughly one prop request on Thursday, ~13 on
Sunday and one on Monday — about 16 a week, plus one game-lines request per
run. The run also logs the API's remaining-credit header so the burn rate is
visible in the job output.

Results are graded into `results/touchdown_log.csv` and
`results/touchdowns/<date>.json`, kept completely separate from the HR and
strikeout logs. Play-by-play lands a day or two after the games, so the
workflow re-grades the last three days rather than only yesterday.

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
  data/nfl.py        ← NFL: schedule, rosters, play-by-play usage (nflverse)
  model/touchdowns.py← anytime-TD math
  run_touchdowns.py  ← NFL entry point → web/touchdowns.json
  calibrate_td.py    ← walk-forward calibration for the TD model
web/                 ← index.html dashboard (vanilla JS + vendored Chart.js)
backtest/backtest.py ← season replay: calibration, ROI, CLV
.github/workflows/daily.yml, touchdowns.yml
```
