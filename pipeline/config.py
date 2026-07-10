"""Every tunable knob in the model lives here.

The model is multiplicative:

    p_PA = LEAGUE_HR_PA * B * P * K * W          (per-plate-appearance HR prob)
    P(>=1 HR) = 1 - (1 - p_PA) ** E[PA | lineup slot]

where B/P/K/W are the batter, pitcher, park, and weather factors
(1.0 = league neutral).  Each factor is computed in pipeline/model/factors.py;
the constants below control how much each input moves the needle.
"""

# ---------------------------------------------------------------- league
# Fallback if we can't compute the live league rate. ~2023-2025 MLB level.
LEAGUE_HR_PA_FALLBACK = 0.032
LEAGUE_BRL_PA_FALLBACK = 0.055   # barrels per PA
LEAGUE_HR_FB_FALLBACK = 0.125    # HR per fly ball
LEAGUE_XISO_FALLBACK = 0.155     # xSLG - xBA
LEAGUE_FB_RATE_FALLBACK = 0.245  # fly balls per batted ball (batter or pitcher)

# ---------------------------------------------------------------- batter
# Regression ballast, in PA: observed rate is blended with league average as
#   regressed = (obs * n + league * ballast) / (n + ballast)
# so a hitter with `ballast` PA of data is weighted 50/50 vs. the league.
BATTER_BALLAST_BRL = 120
BATTER_BALLAST_HRFB = 170
BATTER_BALLAST_XISO = 160

# Weights of the three power signals inside the batter factor
# (geometric mean: B = brl_ratio^w1 * hrfb_ratio^w2 * xiso_ratio^w3).
BATTER_W_BRL = 0.45
BATTER_W_HRFB = 0.30
BATTER_W_XISO = 0.25

# Final elasticity: B <- B ** BATTER_ELASTICITY. <1 shrinks extremes.
BATTER_ELASTICITY = 0.95
BATTER_FACTOR_CAP = (0.35, 3.0)

# ---------------------------------------------------------------- pitcher
# Ballast in batters faced (per handedness split).
PITCHER_BALLAST_HRFB = 400
PITCHER_BALLAST_FB = 150

# Pitcher factor = hrfb_ratio^W_HRFB * fb_ratio^W_FB, then ^ELASTICITY.
# Pitchers control HR less than batters do, hence the stronger shrink.
PITCHER_W_HRFB = 0.55
PITCHER_W_FB = 0.45
PITCHER_ELASTICITY = 0.70
PITCHER_FACTOR_CAP = (0.55, 1.9)
# Share of the batter's expected PA that come against the starter; the rest
# are against the opposing bullpen (see the bullpen factor below).
STARTER_PA_SHARE = 0.62

# ---------------------------------------------------------------- bullpen
# The ~(1 - STARTER_PA_SHARE) of PA a hitter takes against the opposing
# bullpen used to be scored league-neutral.  Now that portion carries the
# opposing bullpen's own HR-vulnerability this season, plus a fatigue bump
# from heavy recent usage (a gassed pen leans on tired / lower-leverage arms).
BULLPEN_BALLAST_FB = 200        # fly balls of sample to trust the pen's HR/FB
BULLPEN_HRFB_ELASTICITY = 0.60  # relievers control HR less than their rate implies
# Fatigue: relief batters-faced over the team's last 3 games vs a typical
# baseline; each extra BF above baseline nudges the factor up a hair.
BULLPEN_BF_BASELINE_3G = 42     # ~14 relief BF/game x 3 games
BULLPEN_FATIGUE_COEF = 0.0035   # multiplier per relief-BF above baseline
BULLPEN_FATIGUE_CAP = (0.95, 1.15)
BULLPEN_FACTOR_CAP = (0.60, 1.70)

# ------------------------------------------------------------- game lines
# Sportsbook total (O/U) and run-line spread encode the market's view of the
# scoring environment.  Implied team runs = total/2 - spread/2 (a favored
# team's spread is negative, so it implies MORE runs).
# CAUTION: the total already reflects park, weather, and starter quality —
# which the model scores separately — so this coefficient is deliberately
# small to avoid double-counting.  Its real job is to capture what the market
# knows that the box-score inputs don't (bullpen state, injuries, umpire,
# sharp money).  Applies to the whole game (starter + bullpen PA).
LEAGUE_AVG_TEAM_RUNS = 4.5
TOTAL_RUNS_COEF = 0.045         # multiplier per implied team-run above avg
TOTAL_FACTOR_CAP = (0.88, 1.15)
ODDS_GAME_MARKETS = "totals,spreads"

# ------------------------------------------------------- batter vs pitcher
# Career HR rate of THIS batter vs THIS starter (MLB StatsAPI vsPlayer).
# HEAVILY regressed toward neutral: BvP samples are tiny (usually < 25 PA)
# and the sabermetric consensus (Tango/Lichtman/Dolphin, "The Book", and
# every streakiness study since) is that BvP history carries almost no
# predictive signal beyond the platoon split and overall quality the model
# already has.  So this is a documented *nudge*, not a driver: a 0-for-9
# should move the number by a couple percent at most.  It applies only to the
# ~STARTER_PA_SHARE of PA faced by the starter.
BVP_MIN_PA = 4              # ignore matchups thinner than this
BVP_BALLAST_PA = 100        # 100 PA of history to be trusted 50/50 vs league
BVP_ELASTICITY = 0.50       # extra shrink on top of the regression
BVP_FACTOR_CAP = (0.85, 1.25)

# ---------------------------------------------------------------- park
# Park factors are indexed 100 = neutral and handedness-specific
# (see pipeline/stadiums.py).  Elasticity <1 because single-season park
# factor tables are noisy.
PARK_ELASTICITY = 0.85

# ---------------------------------------------------------------- weather
# Carry improves with temperature: ~ +0.8% HR probability per degF above 70.
WEATHER_TEMP_REF_F = 70.0
WEATHER_TEMP_COEF = 0.008          # multiplier slope per degF
# Wind blowing straight out to CF: ~ +1.0% HR probability per mph of the
# outward component (negative component = blowing in).
WEATHER_WIND_COEF = 0.010
WEATHER_WIND_CAP_MPH = 18.0        # clamp the |outward component|
# Retractable roofs are usually closed in bad weather; damp the effect.
RETRACTABLE_DAMP = 0.5
# In extreme heat/cold the roof is almost certainly shut (AC/heating), so
# treat the game as indoors: weather factor = 1.0 exactly.
RETRACTABLE_CLOSED_ABOVE_F = 88.0
RETRACTABLE_CLOSED_BELOW_F = 45.0
WEATHER_FACTOR_CAP = (0.70, 1.45)

# ---------------------------------------------------------------- playing time
# Expected PA by lineup slot (1-9). Slot 1 sees ~4.65 PA/game, each slot
# down costs ~0.11 PA.
def expected_pa_for_slot(slot: int) -> float:
    return 4.65 - 0.115 * (max(1, min(9, slot)) - 1)

# ---------------------------------------------------------------- output
PER_PA_PROB_CAP = 0.20     # sanity cap on p_PA
# The JSON carries the union of (top N by probability) and (top N by EV) so
# the dashboard can toggle between the two rankings.
TOP_N = 20

# ---------------------------------------------------------------- odds
ODDS_SPORT_KEY = "baseball_mlb"
ODDS_MARKET = "batter_home_runs"   # Over/Under 0.5 HR props
# When a book lists only the Over side, we can't de-vig the pair; assume this
# overround on the missing side instead.
ASSUMED_SINGLE_SIDE_OVERROUND = 1.06
