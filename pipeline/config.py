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


# ================================================================
# PITCHER STRIKEOUT MODEL  (separate board; independent of the HR model)
# ================================================================
# We project a starter's strikeout total for tonight as
#
#     k_rate  = log5(pitcher_K%, opponent_lineup_K%, league_K%)
#     lambda  = k_rate * expected_batters_faced
#     P(Over line) = NegBinomial_survival(ceil(line); mean=lambda, var=phi*lambda)
#
# log5 is the standard sabermetric way to combine a pitcher rate and a batter
# rate against the league baseline; the negative-binomial (rather than plain
# Poisson) captures the extra game-to-game variance in K totals, which comes
# largely from how deep the starter goes.

LEAGUE_K_PA_FALLBACK = 0.222       # league strikeouts per plate appearance

# Regression ballasts (sample of batters-faced / PA to trust a rate 50/50).
K_PITCHER_BALLAST_PA = 350         # a starter's own K%
K_BATTER_BALLAST_PA = 200          # an opposing hitter's K%
K_PITCHER_FACTOR_CAP = (0.55, 1.9)
K_OPP_FACTOR_CAP = (0.78, 1.28)

# Expected batters faced = regressed (season TBF / start), toward a league
# starter baseline, with a ballast in *starts*.
K_TBF_LEAGUE_START = 23.0
K_TBF_BALLAST = 4.0
K_TBF_CAP = (14.0, 28.0)

# Negative-binomial overdispersion: var = phi * mean (phi > 1 widens the tails
# vs Poisson). ~1.2 matches observed start-level K variance.
K_VARIANCE_INFLATION = 1.20
K_LAMBDA_CAP = (2.0, 14.0)

# Only project starters whose game's first pitch is at/after this ET hour.
K_MIN_START_ET_HOUR = 17           # 5pm ET
K_TOP_N = 20
ODDS_K_MARKET = "pitcher_strikeouts"   # Over/Under total strikeouts


# ================================================================
# GAME LINES MODEL (moneylines + totals)  — separate board
# ================================================================
# Expected runs per team:
#
#   lambda = league_RPG * offense * opp_pitching * park * home_boost
#
#   offense       = team runs/game vs league, regressed by games played
#   opp_pitching  = starter_share * starter_factor + (1-share) * bullpen_factor
#                   where each factor is a FIP-style index (K%, BB%+HBP%, HR%)
#                   mapped linearly to a runs-allowed ratio
#   park          = runs park factor computed from this season's games at the
#                   venue, regressed (NOT the HR park factor)
#   home_boost    = small home-field runs bump
#
# Each team's runs are modeled as a negative binomial (overdispersed Poisson);
# the two distributions give P(win) by direct summation (ties -> extra
# innings, allocated by relative strength) and P(total > line) by convolution.
#
# HONESTY NOTE: moneylines/totals are the most efficient MLB markets. This
# model exists to show *where* it disagrees with Vegas and by how much — a
# large edge here is more likely missing information (injury, lineup,
# bullpen availability) than free money, and the UI says so.

GL_LEAGUE_RPG_FALLBACK = 4.5       # runs per team per game
GL_OFFENSE_BALLAST_G = 45          # games for 50/50 trust in team runs/game
GL_OFFENSE_CAP = (0.75, 1.30)

# FIP-style pitching index, per PA: fip_pa = (13*HR + 3*(BB+HBP) - 2*K) / PA.
# Regressed toward league by PA, then mapped to a runs-allowed ratio:
#   ratio = 1 + GL_FIP_SLOPE * (fip_pa - league_fip_pa)
# Calibration: an ace (K34%, BB6%, HR2.2%) -> ~0.6; a replacement arm
# (K15%, BB11%, HR4%) -> ~1.27.
GL_FIP_SLOPE = 0.9
GL_STARTER_BALLAST_PA = 400
GL_BULLPEN_BALLAST_PA = 700
GL_STARTER_FACTOR_CAP = (0.55, 1.55)
GL_BULLPEN_FACTOR_CAP = (0.75, 1.30)
GL_STARTER_SHARE = 0.58            # starters throw ~58% of a game

GL_PARK_BALLAST_G = 30             # games at the venue for 50/50 trust
GL_PARK_CAP = (0.80, 1.30)
# Home-field advantage, split across both lambdas and calibrated so two equal
# teams give the home side ~53% (the long-run MLB HFA).
GL_HOME_BOOST = 1.033
GL_AWAY_MALUS = 0.968

# Runs are overdispersed: var(team runs) ~ 1.9 * mean at the game level.
GL_VARIANCE_INFLATION = 1.9
GL_LAMBDA_CAP = (2.2, 8.5)
GL_MAX_RUNS = 30                   # pmf truncation per team

ODDS_GL_MARKETS = "h2h,totals"     # one cheap request covers every game
