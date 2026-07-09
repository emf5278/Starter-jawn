"""The multiplicative factors.  Each returns {"value": float, ...breakdown}.

All factors are ratios to league average, so 1.0 = neutral and the final
per-PA probability is simply

    p_PA = league_HR/PA * batter * pitcher * park * weather

Every raw rate is first *regressed to the mean* with a sample-size ballast:

    regressed = (obs * n + league * ballast) / (n + ballast)

i.e. a player with `ballast` worth of sample is trusted 50/50 vs. the league.
This is what keeps a 40-PA hot streak from printing a 3x factor.
"""

from __future__ import annotations

import math

from .. import config


def regress(obs: float | None, n: float | None, league: float, ballast: float) -> float:
    """Shrink an observed rate toward the league mean by sample size."""
    if obs is None or n is None or not math.isfinite(obs) or not math.isfinite(n) or n <= 0:
        return league
    return (obs * n + league * ballast) / (n + ballast)


def _clamp(x: float, lo_hi: tuple[float, float]) -> float:
    return max(lo_hi[0], min(lo_hi[1], x))


# ------------------------------------------------------------------ batter

def batter_power_factor(stats: dict, league: dict) -> dict:
    """B = brl_ratio^0.45 * hrfb_ratio^0.30 * xiso_ratio^0.25, then ^elasticity.

    `stats` carries (rate, sample) pairs from pipeline.data.statcast:
    brl_pa/brl_n, hr_fb/hr_fb_n, xiso/xiso_n.  A missing signal contributes a
    neutral ratio of 1.0 (the regression collapses to the league mean).
    """
    parts = {}
    ratios = []
    for key, n_key, ballast, weight in (
        ("brl_pa", "brl_n", config.BATTER_BALLAST_BRL, config.BATTER_W_BRL),
        ("hr_fb", "hr_fb_n", config.BATTER_BALLAST_HRFB, config.BATTER_W_HRFB),
        ("xiso", "xiso_n", config.BATTER_BALLAST_XISO, config.BATTER_W_XISO),
    ):
        reg = regress(stats.get(key), stats.get(n_key), league[key], ballast)
        ratio = reg / league[key] if league[key] > 0 else 1.0
        parts[key] = {"raw": stats.get(key), "regressed": round(reg, 4),
                      "ratio": round(ratio, 3), "weight": weight}
        ratios.append((ratio, weight))

    value = math.prod(r ** w for r, w in ratios) ** config.BATTER_ELASTICITY
    value = _clamp(value, config.BATTER_FACTOR_CAP)
    return {"value": round(value, 3), "components": parts}


# ------------------------------------------------------------------ pitcher

def pitcher_hr_factor(split: dict | None, league: dict) -> dict:
    """P = hrfb_ratio^0.55 * fbrate_ratio^0.45, then ^0.70 (strong shrink).

    `split` is this starter's line vs. the batter's handedness:
    {pa, hr, fb, gb, bip} from raw Statcast, or {hr_fb, fb_pct, TBF} from the
    FanGraphs fallback, or None (unknown starter -> neutral 1.0).

    HR/FB captures "when they lift it, does it leave"; FB rate captures how
    often they allow lift at all (the GB/FB axis).  A ground-ball pitcher
    suppresses both terms.
    """
    if not split:
        return {"value": 1.0, "components": {}, "note": "no starter data; neutral"}

    if "hr" in split:  # raw statcast split
        fb, pa, hr, bip = split.get("fb", 0), split.get("pa", 0), split.get("hr", 0), split.get("bip", 0)
        hr_fb_obs = hr / fb if fb > 0 else None
        fb_rate_obs = fb / bip if bip > 0 else None
        n_fb, n_bip = fb, bip
    else:  # fangraphs overall fallback
        hr_fb_obs = split.get("hr_fb")
        fb_rate_obs = split.get("fb_pct")
        tbf = split.get("TBF", 0) or 0
        n_bip = tbf * 0.67
        n_fb = n_bip * (fb_rate_obs or config.LEAGUE_FB_RATE_FALLBACK)

    hr_fb = regress(hr_fb_obs, n_fb, league["hr_fb"], config.PITCHER_BALLAST_HRFB * 0.25)
    fb_rate = regress(fb_rate_obs, n_bip, league["fb_rate"], config.PITCHER_BALLAST_FB)
    r_hrfb = hr_fb / league["hr_fb"]
    r_fb = fb_rate / league["fb_rate"]

    value = (r_hrfb ** config.PITCHER_W_HRFB * r_fb ** config.PITCHER_W_FB) ** config.PITCHER_ELASTICITY
    value = _clamp(value, config.PITCHER_FACTOR_CAP)
    return {
        "value": round(value, 3),
        "components": {
            "hr_fb": {"regressed": round(hr_fb, 4), "ratio": round(r_hrfb, 3)},
            "fb_rate": {"regressed": round(fb_rate, 4), "ratio": round(r_fb, 3)},
        },
    }


# ------------------------------------------------------------------ park

def park_factor(stadium: dict, batter_hand: str) -> dict:
    """Handedness-specific HR park factor (100 = neutral), damped.

    K = (PF / 100) ^ PARK_ELASTICITY — the elasticity acknowledges that
    published single-park HR factors are noisy.
    """
    pf = stadium["hr_pf_lhb"] if batter_hand == "L" else stadium["hr_pf_rhb"]
    value = (pf / 100.0) ** config.PARK_ELASTICITY
    return {"value": round(value, 3), "park_factor": pf, "hand": batter_hand,
            "park": stadium["name"]}


# ------------------------------------------------------------------ weather

def weather_factor(weather: dict | None, roof: str) -> dict:
    """W = (1 + temp_coef*(T-70)) * (1 + wind_coef*wind_out), damped by roof.

    Temperature: warmer air is less dense; ~+0.8% HR per degF above 70.
    Wind: the component blowing out to CF adds ~1% per mph (blowing in
    subtracts).  Domes are neutral; retractable roofs get half effect since
    they're usually closed in exactly the weather that would matter.
    """
    if roof == "dome" or weather is None:
        return {"value": 1.0, "note": "dome or no forecast"}

    temp_term = config.WEATHER_TEMP_COEF * (weather["temp_f"] - config.WEATHER_TEMP_REF_F)
    wind_out = _clamp(weather["wind_out_mph"],
                      (-config.WEATHER_WIND_CAP_MPH, config.WEATHER_WIND_CAP_MPH))
    wind_term = config.WEATHER_WIND_COEF * wind_out

    damp = config.RETRACTABLE_DAMP if roof == "retractable" else 1.0
    value = (1.0 + temp_term * damp) * (1.0 + wind_term * damp)
    value = _clamp(value, config.WEATHER_FACTOR_CAP)
    return {"value": round(value, 3), "temp_f": weather["temp_f"],
            "wind_mph": weather["wind_mph"],
            "wind_out_mph": round(weather["wind_out_mph"], 1), "roof": roof}
