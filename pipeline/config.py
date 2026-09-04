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


# ================================================================
# NFL ANYTIME-TD MODEL  — separate board (web/touchdowns.json)
# ================================================================
# P(player scores >=1 TD) for every skill player on the day's slate.
# Structure mirrors the HR board: a team-level expectation from the
# sportsbook line, split into rushing/receiving, then divided among the
# players by *usage share*, and finally P = 1 - exp(-lambda).
#
#   implied_points = total/2 - spread/2          (same math as MLB game lines)
#   team_off_TD    = TD_PTS_SLOPE*implied_points + TD_PTS_INTERCEPT
#   team_rush_TD   = team_off_TD * team_rush_share
#   lambda_player  = team_rush_TD * rush_share * opp_rush_factor
#                  + team_rec_TD  * rec_share  * opp_rec_factor
#   P(anytime TD)  = 1 - exp(-lambda_player)
#
# Passing TDs never count toward an anytime-TD prop, so QBs are carried on
# their rushing usage only.
#
# CALIBRATION. Every constant below was fit on nflverse play-by-play rather
# than guessed; the fits are reproducible with pipeline/calibrate_td.py.

# --- league baselines (2025 regular season, 544 team-games) -------------
TD_LEAGUE_RUSH_TD_PER_TEAM_GAME = 0.938
TD_LEAGUE_REC_TD_PER_TEAM_GAME = 1.491
TD_LEAGUE_OFF_TD_PER_TEAM_GAME = 2.428
TD_LEAGUE_RUSH_SHARE = 0.386       # rushing share of offensive TDs
TD_LEAGUE_POINTS_PER_TEAM_GAME = 23.01

# --- implied points -> expected offensive TDs ---------------------------
# OLS on 2025 team-games: off_TD = 0.12479*points - 0.4434 (r = 0.879).
# Affine, not proportional: field goals are a larger share of the scoring
# in low-total games, so a straight ratio overstates TDs for bad offenses.
TD_PTS_SLOPE = 0.12479
TD_PTS_INTERCEPT = -0.4434
TD_TEAM_OFF_TD_CAP = (1.0, 4.5)

# --- team rushing/receiving split ---------------------------------------
# A team's own rush share of its TDs, regressed toward the league 0.386.
TD_TEAM_SPLIT_BALLAST_TD = 25.0    # team offensive TDs for 50/50 trust
TD_TEAM_RUSH_SHARE_CAP = (0.25, 0.55)

# --- player usage shares -------------------------------------------------
# Walk-forward tested on 2024 + 2025 (weeks 8-18, predicting next week's
# scorer from prior-weeks shares only). Single-signal AUCs:
#
#   rushing: inside-5 carry share .706 | carry share .711 | TD share .686
#   receiving: RZ target share .666 | target share .672 | rec TD share .661
#
# Volume beats history. Prior *rushing TD* share adds nothing once carries
# and goal-line carries are in (grid search put its weight at 0), which is
# the standard TD-regression result — last year's touchdowns are mostly
# noise, this year's carries are not. Receiving keeps a little TD-share
# weight because target quality (air yards, alignment) is not otherwise
# captured. Grid-searched on a 0.1 grid over both seasons; the surface is
# flat near the optimum, so these are round numbers, not fitted decimals.
#
# Rushing weights are POSITION-SPECIFIC, because quarterbacks and running
# backs score rushing TDs for completely different reasons. Re-running the
# same walk-forward split by position (2024 + 2025):
#
#              inside-5 share | carry share | TD share |  best blend
#   RB/FB           .681      |    .685     |   .666   |  .40/.60/.00
#   QB              .638      |    .630     |   .630   |  .50/.00/.50
#
# For a back, carries are the signal and last year's TDs add nothing. For a
# quarterback it inverts: carry share is worthless (it is full of scrambles
# and kneel-downs that have nothing to do with the goal line), while TD
# share carries real information about whether this is a team that hands
# him the ball on the one. Pooling the two positions — which is what a
# single weight vector does — systematically underrates goal-line QBs: the
# eight QBs with the highest goal-line carry share scored in 37.5% of their
# weeks against an 18.0% base rate, which is the ~+150/+170 the books
# actually post on them.
TD_RUSH_SHARE_WEIGHTS = {
    "QB": {"goal_line": 0.50, "carries": 0.00, "tds": 0.50},
    "DEFAULT": {"goal_line": 0.40, "carries": 0.60, "tds": 0.00},
}
TD_REC_SHARE_WEIGHTS = {"red_zone": 0.10, "targets": 0.60, "tds": 0.30}

# "Goal line" = carries from inside the 5 (5.6% of carries, 59% of rushing
# TDs). "Red zone" = targets from inside the 20 (14% of targets, 71% of
# receiving TDs).
TD_GOAL_LINE_YARDLINE = 5
TD_RED_ZONE_YARDLINE = 20

# Shrink a player's usage share toward a replacement-level share using his
# opportunity count, then renormalise within the team so the shares of the
# players actually on the roster sum to 1. Ballasts are in opportunities
# (carries / targets) for 50/50 trust.
TD_RUSH_SHARE_BALLAST = 40.0
TD_REC_SHARE_BALLAST = 35.0
# Replacement-level share for a body with no history. Calibrated, not
# guessed: at 0.04 the ~15 zero-usage players on every active roster
# collectively held ~60% of the team's shares, which bled probability off
# the real contributors and onto the bench (the 0.05-0.10 bin predicted
# 0.085 against an observed 0.047). Swept jointly with the concentration
# exponent below on 2025 weeks 8-18; 0.012 is the best setting on both log
# loss and Brier. Getting this bin right matters more than it looks: an
# overpredicted longshot is exactly what turns into a phantom +EV pick.
TD_REPLACEMENT_RUSH_SHARE = 0.012
TD_REPLACEMENT_REC_SHARE = 0.009

# Share concentration. Raw usage shares are *too* concentrated at the top:
# a back with 45% of the goal-line carries does not score 45% of his team's
# rushing TDs, because game script, mid-game injuries and vulture scores all
# spread the work out on the day. Raising every share to this power before
# renormalising compresses the distribution (a < 1 pulls the leaders down
# and the rotation up). Swept on 2025 weeks 8-18: 0.80 is the joint best on
# Brier and log loss, and lands the top bin almost exactly (predicted 0.535
# vs observed 0.528, against 0.556 vs 0.410 with no compression at all).
TD_SHARE_CONCENTRATION = 0.80

# Rookies have no NFL usage at all. Draft capital is the only prior worth
# having: a first-round back is not a replacement-level body. Maps draft
# slot -> a starting usage share, interpolated and flat outside the range.
TD_ROOKIE_DRAFT_PRIOR = {
    "RB": [(1, 0.42), (40, 0.28), (100, 0.14), (200, 0.06)],
    "WR": [(1, 0.20), (40, 0.14), (100, 0.08), (200, 0.04)],
    "TE": [(1, 0.15), (40, 0.10), (100, 0.06), (200, 0.03)],
}
TD_UNDRAFTED_SHARE = 0.02

# --- opponent defence ----------------------------------------------------
# TDs allowed vs league, regressed by games. Deliberately small and hard
# capped: defensive TD-prevention is noisy year to year and mostly already
# priced into the game total this model is anchored on, so letting it swing
# a pick would be double-counting.
TD_DEF_BALLAST_G = 10.0
TD_DEF_FACTOR_CAP = (0.80, 1.25)

# --- prior-season blend / cold start ------------------------------------
# Weight on last season when blending usage with the current season, same
# idea as the HR model. In Week 1 there is no current-season data at all,
# so the board runs entirely on last year -- see the confidence rules.
TD_PRIOR_SEASON_WEIGHT = 0.6

# Confidence. Anytime-TD numbers built from last season's usage are a
# genuinely weaker product than mid-season ones: free agency, the draft and
# training camp all move usage, and none of that is in the data yet. The
# page badges every pick so nobody reads a Week 1 number as a Week 10
# number.
#   high   - enough current-season usage to stand on its own
#   medium - blended, or a player who changed teams in the off-season
#   low    - no current-season usage at all (Week 1-2), or a rookie
TD_CONF_HIGH_MIN_GAMES = 4         # current-season team games played
TD_CONF_MEDIUM_MIN_GAMES = 2

TD_LAMBDA_CAP = (0.005, 1.60)      # P(TD) roughly 0.5% .. 80%
TD_TOP_N = 20
TD_MIN_LAMBDA_TO_LIST = 0.02       # don't rank deep bench bodies

# --- odds ---------------------------------------------------------------
ODDS_NFL_SPORT_KEY = "americanfootball_nfl"
ODDS_TD_MARKET = "player_anytime_td"      # Yes/No, "anytime touchdown scorer"
ODDS_NFL_GAME_MARKETS = "totals,spreads"  # one cheap call -> implied points
