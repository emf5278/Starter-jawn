"""Pitcher strikeout projection and P(Over the line).

Mirrors the HR model's philosophy — transparent, documented, tunable factors —
but the target is a *count* (total Ks) compared to a sportsbook Over/Under line,
not a binary event.

    k_rate = log5(pitcher_K%, opponent_K%, league_K%)   # matchup strikeout rate
    lambda = k_rate * expected_batters_faced              # expected K total
    P(Over line) = NB_survival(ceil(line); mean=lambda, var=phi*lambda)

Every rate is regressed to the league mean by sample size first (same ballast
trick as the HR model), so a two-start hot streak can't run the projection.
"""

from __future__ import annotations

import math

from .. import config


def _clamp(x: float, lo_hi: tuple[float, float]) -> float:
    return max(lo_hi[0], min(lo_hi[1], x))


def regress(obs: float | None, n: float | None, league: float, ballast: float) -> float:
    if (obs is None or n is None or not math.isfinite(obs)
            or not math.isfinite(n) or n <= 0):
        return league
    return (obs * n + league * ballast) / (n + ballast)


# ------------------------------------------------------------------ factors

def pitcher_k_factor(k_rate: float | None, n: float | None, league: dict) -> dict:
    """Starter's strikeout rate (K per batter faced), regressed, vs league."""
    reg = regress(k_rate, n, league["k_pa"], config.K_PITCHER_BALLAST_PA)
    ratio = _clamp(reg / league["k_pa"], config.K_PITCHER_FACTOR_CAP)
    return {"regressed": round(reg, 4), "ratio": round(ratio, 3),
            "raw": None if k_rate is None else round(k_rate, 4), "n": n}


def opponent_k_factor(batters: list[tuple], league: dict) -> dict:
    """Opposing lineup's whiff propensity: PA-weighted mean of each hitter's
    regressed K%.  `batters` is a list of (k_rate, n, weight)."""
    num = den = 0.0
    for kr, n, w in batters:
        num += regress(kr, n, league["k_pa"], config.K_BATTER_BALLAST_PA) * w
        den += w
    opp = num / den if den else league["k_pa"]
    ratio = _clamp(opp / league["k_pa"], config.K_OPP_FACTOR_CAP)
    return {"opp_k_rate": round(opp, 4), "ratio": round(ratio, 3), "n_batters": len(batters)}


def expected_tbf(tbf_per_start: float | None, n_starts: float | None) -> dict:
    """Expected batters faced = regressed season (TBF/start) toward league."""
    reg = regress(tbf_per_start, n_starts, config.K_TBF_LEAGUE_START, config.K_TBF_BALLAST)
    val = _clamp(reg, config.K_TBF_CAP)
    return {"value": round(val, 2), "raw": None if tbf_per_start is None else round(tbf_per_start, 1),
            "starts": n_starts}


# ------------------------------------------------------------------ combine

def _log5(a: float, b: float, lg: float) -> float:
    """Combine a pitcher rate and a batter rate against the league baseline."""
    num = a * b / lg
    den = num + (1 - a) * (1 - b) / (1 - lg)
    return num / den if den > 0 else lg


def expected_strikeouts(pk_regressed: float, opp_regressed: float, tbf: float,
                        league: dict) -> dict:
    k_rate = _log5(pk_regressed, opp_regressed, league["k_pa"])
    lam = _clamp(k_rate * tbf, config.K_LAMBDA_CAP)
    return {"k_rate": round(k_rate, 4), "expected_ks": round(lam, 2)}


# ------------------------------------------------------ count distribution

def _pois_sf(k: int, mean: float) -> float:
    """P(X >= k) for Poisson(mean)."""
    cdf = 0.0
    for j in range(0, int(k)):
        cdf += math.exp(-mean + j * math.log(mean) - math.lgamma(j + 1))
    return min(1.0, max(0.0, 1.0 - cdf))


def _nb_sf(k: int, mean: float, phi: float) -> float:
    """P(X >= k) for a negative binomial with mean `mean` and var = phi*mean."""
    if mean <= 0:
        return 0.0
    if phi <= 1.0:
        return _pois_sf(k, mean)
    r = mean / (phi - 1.0)          # NB "size"
    p = r / (r + mean)              # NB success prob (var = mean + mean^2/r)
    logp, log1mp = math.log(p), math.log1p(-p)
    cdf = 0.0
    for j in range(0, int(k)):
        cdf += math.exp(math.lgamma(j + r) - math.lgamma(r) - math.lgamma(j + 1)
                        + r * logp + j * log1mp)
    return min(1.0, max(0.0, 1.0 - cdf))


def prob_over(line: float, lam: float, phi: float | None = None) -> float:
    """P(strikeouts > line).  For a .5 line, Over means X >= ceil(line)."""
    phi = config.K_VARIANCE_INFLATION if phi is None else phi
    return _nb_sf(math.ceil(line), lam, phi)


def predict_pitcher(pk: dict, opp: dict, tbf: dict, line: float | None,
                    league: dict) -> dict:
    """Full projection for one starter; `line` is the book's K total (or None)."""
    proj = expected_strikeouts(pk["regressed"], opp["opp_k_rate"], tbf["value"], league)
    out = {
        "expected_ks": proj["expected_ks"],
        "k_rate": proj["k_rate"],
        "factors": {
            "league_k_pa": round(league["k_pa"], 4),
            "pitcher": pk, "opponent": opp, "expected_tbf": tbf,
        },
    }
    if line is not None:
        p_over = prob_over(line, proj["expected_ks"])
        out["line"] = line
        out["prob_over"] = round(p_over, 4)
        out["prob_under"] = round(1 - p_over, 4)
    return out
