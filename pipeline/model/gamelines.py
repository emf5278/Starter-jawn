"""Moneyline and totals model: expected runs -> full run distributions.

    lambda_team = league_RPG * offense * opp_pitching * park * home_boost

Each team's runs follow a negative binomial (mean lambda, var = phi*lambda;
runs are strongly overdispersed at the game level).  From the two pmfs:

    P(home wins) = sum_i P(H=i) * P(A < i)  +  P(tie) * lambda_H/(lambda_H+lambda_A)
                   (ties go to extra innings; allocate by relative strength)
    P(total > line) from the convolution of the two pmfs.

Design note — why this can't print "locks": every input here is public season
data, which is exactly what the market has already priced.  The model's job is
to be a clean, independent estimate so *disagreements* with Vegas are visible
and explainable.  Historically, disagreements of a few percent in these markets
are mostly model blindness (injuries, lineup news, bullpen availability, umpire,
travel), which is why the UI flags big edges as suspect rather than as value.
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

def offense_factor(rpg: float | None, games: float | None, league_rpg: float) -> dict:
    """Team runs/game vs league, regressed by games played."""
    reg = regress(rpg, games, league_rpg, config.GL_OFFENSE_BALLAST_G)
    ratio = _clamp(reg / league_rpg, config.GL_OFFENSE_CAP)
    return {"ratio": round(ratio, 3), "rpg": None if rpg is None else round(rpg, 2),
            "games": games}


def _fip_pa(k: float, bb: float, hr: float, pa: float) -> float:
    return (13.0 * hr + 3.0 * bb - 2.0 * k) / pa


def pitching_factor(stats: dict | None, league: dict, ballast: float,
                    cap: tuple[float, float]) -> dict:
    """FIP-style runs-allowed ratio from K%, BB%(+HBP), HR% — regressed.

        fip_pa = (13*HR + 3*BB - 2*K) / PA
        ratio  = 1 + GL_FIP_SLOPE * (regressed fip_pa - league fip_pa)

    `stats` = {pa,k,bb,hr} or None (unknown arm -> league neutral 1.0).
    """
    lg = league["fip_pa"]
    if not stats or not stats.get("pa"):
        return {"ratio": 1.0, "note": "no data; league-average assumed"}
    obs = _fip_pa(stats["k"], stats["bb"], stats["hr"], stats["pa"])
    reg = regress(obs, stats["pa"], lg, ballast)
    ratio = _clamp(1.0 + config.GL_FIP_SLOPE * (reg - lg), cap)
    return {"ratio": round(ratio, 3), "fip_pa": round(obs, 4),
            "regressed": round(reg, 4), "pa": stats["pa"],
            "k_pct": round(stats["k"] / stats["pa"], 3),
            "bb_pct": round(stats["bb"] / stats["pa"], 3),
            "hr_pct": round(stats["hr"] / stats["pa"], 3)}


def park_runs_factor(total_rpg: float | None, games: float | None,
                     league_rpg: float) -> dict:
    """Venue total-runs environment vs league (2*RPG), regressed by games."""
    lg_total = 2.0 * league_rpg
    reg = regress(total_rpg, games, lg_total, config.GL_PARK_BALLAST_G)
    ratio = _clamp(reg / lg_total, config.GL_PARK_CAP)
    return {"ratio": round(ratio, 3),
            "total_rpg": None if total_rpg is None else round(total_rpg, 2),
            "games": games}


def team_lambda(league_rpg: float, offense: dict, starter: dict, bullpen: dict,
                park: dict, is_home: bool) -> float:
    """Expected runs for one team tonight."""
    opp_pitching = (config.GL_STARTER_SHARE * starter["ratio"]
                    + (1 - config.GL_STARTER_SHARE) * bullpen["ratio"])
    lam = (league_rpg * offense["ratio"] * opp_pitching * park["ratio"]
           * (config.GL_HOME_BOOST if is_home else config.GL_AWAY_MALUS))
    return _clamp(lam, config.GL_LAMBDA_CAP)


# ------------------------------------------------------ run distributions

def nb_pmf(mean: float, phi: float, nmax: int) -> list[float]:
    """pmf of a negative binomial with var = phi*mean, truncated+renormalized."""
    if phi <= 1.0:  # degenerate to Poisson
        pmf = [math.exp(-mean + k * math.log(mean) - math.lgamma(k + 1))
               for k in range(nmax + 1)]
    else:
        r = mean / (phi - 1.0)
        p = r / (r + mean)
        logp, log1mp = math.log(p), math.log1p(-p)
        pmf = [math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                        + r * logp + k * log1mp) for k in range(nmax + 1)]
    s = sum(pmf)
    return [x / s for x in pmf]


def win_probability(lam_home: float, lam_away: float,
                    phi: float | None = None) -> dict:
    """P(home wins): direct summation over the two pmfs; ties (extra innings)
    allocated by relative lambda."""
    phi = config.GL_VARIANCE_INFLATION if phi is None else phi
    n = config.GL_MAX_RUNS
    h = nb_pmf(lam_home, phi, n)
    a = nb_pmf(lam_away, phi, n)
    cdf_a = []
    run = 0.0
    for x in a:
        cdf_a.append(run)   # P(A < i) exclusive
        run += x
    p_home_lead = sum(h[i] * cdf_a[i] for i in range(n + 1))
    p_tie = sum(h[i] * a[i] for i in range(n + 1))
    share = lam_home / (lam_home + lam_away)
    p_home = p_home_lead + p_tie * share
    return {"p_home": round(p_home, 4), "p_away": round(1 - p_home, 4),
            "p_tie_reg": round(p_tie, 4)}


def total_distribution(lam_home: float, lam_away: float,
                       phi: float | None = None) -> list[float]:
    """pmf of home+away runs (independent-team approximation)."""
    phi = config.GL_VARIANCE_INFLATION if phi is None else phi
    n = config.GL_MAX_RUNS
    h = nb_pmf(lam_home, phi, n)
    a = nb_pmf(lam_away, phi, n)
    out = [0.0] * (2 * n + 1)
    for i, hi in enumerate(h):
        for j, aj in enumerate(a):
            out[i + j] += hi * aj
    return out


def total_probs(line: float, lam_home: float, lam_away: float,
                phi: float | None = None) -> dict:
    """{p_over, p_under, p_push} for a totals line.

    Over wins strictly above the line; integer lines can push (total == line),
    which returns the stake — the EV math must treat it as neither win nor loss.
    """
    pmf = total_distribution(lam_home, lam_away, phi)
    k_over = math.floor(line) + 1          # first total that wins the Over
    p_over = sum(pmf[k_over:])
    p_push = pmf[int(line)] if float(line).is_integer() and line <= len(pmf) - 1 else 0.0
    p_under = max(0.0, 1.0 - p_over - p_push)
    return {"p_over": round(p_over, 4), "p_under": round(p_under, 4),
            "p_push": round(p_push, 4)}
