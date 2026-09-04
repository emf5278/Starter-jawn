"""Anytime-TD model: P(player scores >=1 rushing or receiving TD today).

The shape is deliberately the same as the HR model — a league-anchored
expectation, adjusted by a handful of heavily-regressed factors, each of
which you can read off the "why" panel and disagree with.

    team_off_TD   = f(implied points from the game line)
    team_rush_TD  = team_off_TD * team_rush_share
    lambda        = team_rush_TD * rush_share * opp_rush_factor
                  + team_rec_TD  * rec_share  * opp_rec_factor
    P(anytime TD) = 1 - exp(-lambda)

The two share terms are where the work is.  A player's share of his team's
rushing TDs is driven by *volume*, not by last year's touchdowns: carries and
goal-line carries predict next week's scorer materially better than prior TD
share does (see the AUCs in config.py).  Shares are shrunk toward a
replacement-level body by opportunity count and then renormalised across the
players actually on the roster, so a departed starter's work is redistributed
to the players who inherit it rather than vanishing.

Poisson, not binomial: a player can score twice, and P(>=1) = 1 - exp(-lambda)
handles that cleanly.  Multi-TD games are ~13% of scoring games, so ignoring
them would bias every probability upward.
"""

from __future__ import annotations

import math

from .. import config


def _cap(x: float, lo_hi: tuple[float, float]) -> float:
    return max(lo_hi[0], min(lo_hi[1], x))


def regress(obs: float, n: float, prior: float, ballast: float) -> float:
    """(obs*n + prior*ballast) / (n + ballast) — the same shrinkage the HR
    model uses everywhere."""
    if n is None or n <= 0:
        return prior
    return (obs * n + prior * ballast) / (n + ballast)


# --------------------------------------------------------------- team level

def expected_team_tds(implied_points: float | None) -> dict:
    """Implied points -> expected offensive (rush + rec) TDs.

    Affine rather than proportional because field goals make up a bigger
    share of the scoring in low-total games.
    """
    if implied_points is None:
        return {"value": config.TD_LEAGUE_OFF_TD_PER_TEAM_GAME,
                "implied_points": None, "source": "league average"}
    raw = config.TD_PTS_SLOPE * implied_points + config.TD_PTS_INTERCEPT
    return {"value": round(_cap(raw, config.TD_TEAM_OFF_TD_CAP), 4),
            "implied_points": round(implied_points, 2), "source": "game line"}


def team_rush_share(team_rush_tds: float, team_off_tds: float) -> dict:
    """How this team's TDs split between the run and the pass, regressed
    toward the league's 38.6% rushing."""
    obs = (team_rush_tds / team_off_tds) if team_off_tds else config.TD_LEAGUE_RUSH_SHARE
    val = regress(obs, team_off_tds, config.TD_LEAGUE_RUSH_SHARE,
                  config.TD_TEAM_SPLIT_BALLAST_TD)
    return {"value": round(_cap(val, config.TD_TEAM_RUSH_SHARE_CAP), 4),
            "observed": round(obs, 4), "n_tds": round(team_off_tds, 1)}


def opponent_factor(tds_allowed: float, games: float, league_per_game: float) -> dict:
    """Opponent TDs allowed vs league, regressed by games and hard capped.

    Small on purpose: most of a defence's quality is already inside the game
    total this model is anchored on, so a big swing here double-counts it.
    """
    if not games:
        return {"value": 1.0, "per_game": None, "n_games": 0}
    per_game = tds_allowed / games
    reg = regress(per_game, games, league_per_game, config.TD_DEF_BALLAST_G)
    val = reg / league_per_game if league_per_game else 1.0
    return {"value": round(_cap(val, config.TD_DEF_FACTOR_CAP), 4),
            "per_game": round(per_game, 3), "n_games": round(games, 1)}


# ------------------------------------------------------------- player level

def _draft_prior(position: str, draft_number: float | None) -> float:
    """Rookies have no usage; draft capital is the only prior worth having."""
    table = config.TD_ROOKIE_DRAFT_PRIOR.get(
        position if position in config.TD_ROOKIE_DRAFT_PRIOR else "WR")
    if draft_number is None or not (draft_number == draft_number):  # NaN-safe
        return config.TD_UNDRAFTED_SHARE
    pts = sorted(table)
    if draft_number <= pts[0][0]:
        return pts[0][1]
    if draft_number >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= draft_number <= x1:
            t = (draft_number - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return config.TD_UNDRAFTED_SHARE


def raw_rush_share(player: dict, team: dict, position: str = "") -> tuple[float, float]:
    """Blended rushing usage share, and the opportunity count behind it.

    Weights are position-specific: a quarterback's carry share is mostly
    scrambles, so his goal-line work and TD history carry the signal, while
    for a back it is the other way round (see config for the AUCs).
    """
    w = config.TD_RUSH_SHARE_WEIGHTS.get(position, config.TD_RUSH_SHARE_WEIGHTS["DEFAULT"])
    carries, gl, tds = team.get("carries", 0), team.get("goal_line_carries", 0), team.get("rush_tds", 0)
    share = (
        w["goal_line"] * (player.get("goal_line_carries", 0) / gl if gl else 0.0)
        + w["carries"] * (player.get("carries", 0) / carries if carries else 0.0)
        + w["tds"] * (player.get("rush_tds", 0) / tds if tds else 0.0)
    )
    return share, float(player.get("carries", 0))


def raw_rec_share(player: dict, team: dict) -> tuple[float, float]:
    w = config.TD_REC_SHARE_WEIGHTS
    tgt, rz, tds = team.get("targets", 0), team.get("red_zone_targets", 0), team.get("rec_tds", 0)
    share = (
        w["red_zone"] * (player.get("red_zone_targets", 0) / rz if rz else 0.0)
        + w["targets"] * (player.get("targets", 0) / tgt if tgt else 0.0)
        + w["tds"] * (player.get("rec_tds", 0) / tds if tds else 0.0)
    )
    return share, float(player.get("targets", 0))


def shrink_share(raw: float, opportunities: float, position: str,
                 draft_number: float | None, is_rookie: bool, kind: str) -> float:
    """Shrink toward a replacement-level (or draft-implied) share."""
    if kind == "rush":
        prior = config.TD_REPLACEMENT_RUSH_SHARE
        ballast = config.TD_RUSH_SHARE_BALLAST
    else:
        prior = config.TD_REPLACEMENT_REC_SHARE
        ballast = config.TD_REC_SHARE_BALLAST
    if is_rookie and opportunities <= 0:
        # No NFL snaps at all: the draft prior *is* the estimate.
        dp = _draft_prior(position, draft_number)
        return dp if kind == "rush" and position in ("RB", "FB") else (
            dp if kind == "rec" and position in ("WR", "TE") else prior)
    return regress(raw, opportunities, prior, ballast)


def normalise(shares: dict[str, float]) -> dict[str, float]:
    """Renormalise so the roster's shares sum to 1.

    This is what redistributes a departed starter's carries to whoever is
    actually on the roster now, instead of leaving the team short.

    Shares are first raised to TD_SHARE_CONCENTRATION, which compresses the
    distribution: raw usage over-concentrates at the top relative to who
    actually scores on a given Sunday.
    """
    a = config.TD_SHARE_CONCENTRATION
    adj = {k: (v ** a if v > 0 else 0.0) for k, v in shares.items()}
    total = sum(adj.values())
    if total <= 0:
        return {k: 0.0 for k in shares}
    return {k: v / total for k, v in adj.items()}


def confidence(current_season_games: float, changed_teams: bool, is_rookie: bool) -> str:
    if is_rookie or current_season_games < config.TD_CONF_MEDIUM_MIN_GAMES:
        return "low"
    if changed_teams or current_season_games < config.TD_CONF_HIGH_MIN_GAMES:
        return "medium"
    return "high"


def predict_player(team_off_td: dict, rush_split: dict, rush_share: float,
                   rec_share: float, opp_rush: dict, opp_rec: dict) -> dict:
    """Combine into lambda and P(>=1 TD)."""
    off = team_off_td["value"]
    team_rush = off * rush_split["value"]
    team_rec = off * (1.0 - rush_split["value"])

    lam_rush = team_rush * rush_share * opp_rush["value"]
    lam_rec = team_rec * rec_share * opp_rec["value"]
    lam = _cap(lam_rush + lam_rec, config.TD_LAMBDA_CAP)
    return {
        "lambda": round(lam, 5),
        "lambda_rush": round(lam_rush, 5),
        "lambda_rec": round(lam_rec, 5),
        "prob": round(1.0 - math.exp(-lam), 5),
        "expected_tds": round(lam, 4),
        "team_expected_off_tds": round(off, 3),
        "team_expected_rush_tds": round(team_rush, 3),
        "team_expected_rec_tds": round(team_rec, 3),
    }
