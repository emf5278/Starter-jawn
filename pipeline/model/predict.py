"""Combine the factors into P(player hits >=1 HR today).

    p_starter = cap(league_HR/PA * B * P_starter * K * W)
    p_bullpen = cap(league_HR/PA * B * 1.0       * K * W)

    n_starter = E[PA | slot] * STARTER_PA_SHARE
    n_bullpen = E[PA | slot] * (1 - STARTER_PA_SHARE)

    P(>=1 HR) = 1 - (1 - p_starter)^n_starter * (1 - p_bullpen)^n_bullpen

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
) -> dict:
    b = F.batter_power_factor(batter_stats, league)
    p = F.pitcher_hr_factor(pitcher_split, league)
    k = F.park_factor(stadium, batter_hand)
    w = F.weather_factor(weather, stadium["roof"])

    base = league["hr_pa"]
    common = base * b["value"] * k["value"] * w["value"]
    p_starter = min(config.PER_PA_PROB_CAP, common * p["value"])
    p_bullpen = min(config.PER_PA_PROB_CAP, common)

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
            "expected_pa": {"value": round(epa, 2), "lineup_slot": lineup_slot,
                            "starter_share": config.STARTER_PA_SHARE},
        },
    }


def ev_per_dollar(model_prob: float, decimal_odds: float) -> float:
    """Flat $1 stake: EV = p*(d-1) - (1-p)."""
    return model_prob * (decimal_odds - 1) - (1 - model_prob)
