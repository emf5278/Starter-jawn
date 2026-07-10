"""Combine the factors into P(player hits >=1 HR today).

    common    = league_HR/PA * B * K * W * T          (whole-game context)
    p_starter = cap(common * P_starter * BvP)          (faces the starter)
    p_bullpen = cap(common * Pen)                       (faces the bullpen)

    n_starter = E[PA | slot] * STARTER_PA_SHARE
    n_bullpen = E[PA | slot] * (1 - STARTER_PA_SHARE)

    P(>=1 HR) = 1 - (1 - p_starter)^n_starter * (1 - p_bullpen)^n_bullpen

where each factor slots in at its natural scope:
    B   batter power          — every PA
    K   park (handedness)      — every PA
    W   weather / roof         — every PA
    T   game total/spread      — every PA (market's scoring-environment view)
    P   starter HR-vuln        — starter PA only
    BvP career vs this starter — starter PA only (heavily regressed)
    Pen opposing-bullpen HR-vuln x fatigue — bullpen PA only

The exponent form treats each PA as an independent Bernoulli trial with the
same per-PA probability — a simplification, but a good one at these scales.
"""

from __future__ import annotations

from .. import config
from . import factors as F


def predict_player(
    batter_stats: dict,
    pitcher_split: dict | None,
    stadium: dict,
    weather: dict | None,
    batter_hand: str,
    lineup_slot: int,
    league: dict,
    *,
    bvp: dict | None = None,
    bullpen: dict | None = None,
    game_line: dict | None = None,
) -> dict:
    b = F.batter_power_factor(batter_stats, league)
    p = F.pitcher_hr_factor(pitcher_split, league)
    k = F.park_factor(stadium, batter_hand)
    w = F.weather_factor(weather, stadium["roof"])
    v = F.bvp_factor(bvp, league)
    pen = F.bullpen_factor(bullpen, league)
    t = F.game_total_factor(game_line)

    base = league["hr_pa"]
    common = base * b["value"] * k["value"] * w["value"] * t["value"]
    p_starter = min(config.PER_PA_PROB_CAP, common * p["value"] * v["value"])
    p_bullpen = min(config.PER_PA_PROB_CAP, common * pen["value"])

    epa = config.expected_pa_for_slot(lineup_slot)
    n_starter = epa * config.STARTER_PA_SHARE
    n_bullpen = epa * (1 - config.STARTER_PA_SHARE)
    prob = 1 - (1 - p_starter) ** n_starter * (1 - p_bullpen) ** n_bullpen

    return {
        "prob": round(prob, 4),
        "per_pa_prob_vs_starter": round(p_starter, 4),
        "expected_pa": round(epa, 2),
        "factors": {
            "league_hr_pa": round(base, 4),
            "batter": b,
            "pitcher": p,
            "park": k,
            "weather": w,
            "bvp": v,
            "bullpen": pen,
            "game_total": t,
            "expected_pa": {"value": round(epa, 2), "lineup_slot": lineup_slot,
                            "starter_share": config.STARTER_PA_SHARE},
        },
    }


def ev_per_dollar(model_prob: float, decimal_odds: float) -> float:
    """Flat $1 stake: EV = p*(d-1) - (1-p)."""
    return model_prob * (decimal_odds - 1) - (1 - model_prob)
