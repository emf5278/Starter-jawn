"""Batter power stats, pitcher HR-vulnerability splits, and league baselines.

Sources (all via pybaseball, no API key):
  * Baseball Savant leaderboards: barrels/PA, xBA/xSLG (-> xISO)
  * FanGraphs season stats: HR/FB, FB%, PA
  * Raw Statcast events: pitcher handedness splits (HR/FB, GB/FB vs LHB/RHB)

Everything returns plain pandas DataFrames keyed by MLBAM player id, with
per-metric sample sizes so the model can regress each rate to the mean.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Weight applied to last season's sample when blending with season-to-date.
PREV_SEASON_WEIGHT = 0.6


def _as_fraction(s: pd.Series) -> pd.Series:
    """FanGraphs/Savant rate columns are sometimes 0-1, sometimes 0-100."""
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().median() > 1.0:
        s = s / 100.0
    return s


# --------------------------------------------------------------------- batters

def _batter_season_stats(season: int) -> pd.DataFrame:
    """One season of power inputs per batter: brl_pa, xiso, hr_fb + samples."""
    from pybaseball import (
        batting_stats,
        playerid_reverse_lookup,
        statcast_batter_exitvelo_barrels,
        statcast_batter_expected_stats,
    )

    barrels = statcast_batter_exitvelo_barrels(season, minBBE=20)
    barrels = barrels.rename(columns={"attempts": "bbe"})
    barrels["brl_pa"] = _as_fraction(barrels["brl_pa"])
    barrels = barrels[["player_id", "brl_pa", "bbe"]]

    xstats = statcast_batter_expected_stats(season, minPA=25)
    xstats["xiso"] = pd.to_numeric(xstats["est_slg"], errors="coerce") - pd.to_numeric(
        xstats["est_ba"], errors="coerce"
    )
    xstats = xstats.rename(columns={"pa": "xiso_pa"})[["player_id", "xiso", "xiso_pa"]]

    # FanGraphs blocks requests from cloud/CI IPs (403). HR/FB from here is a
    # bonus; when unavailable it gets filled from raw Statcast events instead
    # (see batter_hrfb_from_events) or regresses to the league mean.
    try:
        fg = batting_stats(season, qual=30)[["IDfg", "Name", "PA", "HR", "HR/FB", "FB%"]]
        fg["hr_fb"] = _as_fraction(fg["HR/FB"])
        fg["fb_pct"] = _as_fraction(fg["FB%"])
        # approximate fly-ball count for the regression ballast:
        # PA * (balls in play share ~0.67) * FB%
        fg["fb_n"] = fg["PA"] * 0.67 * fg["fb_pct"]
        ids = playerid_reverse_lookup(fg["IDfg"].tolist(), key_type="fangraphs")
        fg = fg.merge(
            ids[["key_fangraphs", "key_mlbam"]],
            left_on="IDfg",
            right_on="key_fangraphs",
            how="inner",
        ).rename(columns={"key_mlbam": "player_id"})
        fg = fg[["player_id", "PA", "hr_fb", "fb_n"]].rename(columns={"PA": "pa"})
    except Exception:
        log.warning("FanGraphs batting stats unavailable for %s (blocked/down); "
                    "HR/FB will come from raw Statcast or regress to league",
                    season, exc_info=True)
        fg = pd.DataFrame(columns=["player_id", "pa", "hr_fb", "fb_n"])

    out = barrels.merge(xstats, on="player_id", how="outer").merge(
        fg, on="player_id", how="outer"
    )
    out["player_id"] = out["player_id"].astype("Int64")
    return out


def batter_power_stats(season: int) -> pd.DataFrame:
    """Blend season-to-date with last season (downweighted) so April isn't chaos.

    Rates are combined as sample-weighted means with the previous season's
    sample multiplied by PREV_SEASON_WEIGHT.
    """
    cur = _batter_season_stats(season)
    try:
        prev = _batter_season_stats(season - 1)
    except Exception:
        log.warning("no previous-season batter stats; using current only", exc_info=True)
        prev = pd.DataFrame(columns=cur.columns)

    merged = cur.merge(prev, on="player_id", how="outer", suffixes=("", "_prev"))

    def blend(rate: str, n: str) -> tuple[pd.Series, pd.Series]:
        r0 = merged.get(rate)
        n0 = merged.get(n).fillna(0)
        r1 = merged.get(f"{rate}_prev")
        n1 = merged.get(f"{n}_prev", pd.Series(0, index=merged.index)).fillna(0) * PREV_SEASON_WEIGHT
        num = (r0.fillna(0) * n0) + (r1.fillna(0) * n1)
        den = n0.where(r0.notna(), 0) + n1.where(r1.notna(), 0)
        return num / den.replace(0, np.nan), den

    out = pd.DataFrame({"player_id": merged["player_id"]})
    out["brl_pa"], out["brl_n"] = blend("brl_pa", "bbe")
    out["xiso"], out["xiso_n"] = blend("xiso", "xiso_pa")
    out["hr_fb"], out["hr_fb_n"] = blend("hr_fb", "fb_n")
    return out.set_index("player_id")


# ---------------------------------------------------------------- raw events

def season_events(season: int, end_date: dt.date) -> pd.DataFrame | None:
    """Season-to-date raw Statcast events (regular season only), or None.

    pybaseball caches the underlying daily chunks, so repeat runs are
    incremental. One download feeds pitcher splits, batter HR/FB, and
    league rates.
    """
    from pybaseball import statcast

    try:
        ev = statcast(start_dt=f"{season}-03-15",
                      end_dt=end_date.strftime("%Y-%m-%d"), verbose=False)
    except Exception:
        log.warning("raw statcast pull failed", exc_info=True)
        return None
    if ev is None or ev.empty:
        return None
    return ev[ev["game_type"] == "R"] if "game_type" in ev.columns else ev


def _pa_flags(ev: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row per plate appearance with hr/fb/gb/bip flags."""
    pa_end = ev["events"].notna() & (ev["events"] != "")
    df = ev.loc[pa_end, cols + ["events", "bb_type"]].copy()
    df["hr"] = (df["events"] == "home_run").astype(int)
    df["fb"] = df["bb_type"].isin(["fly_ball", "popup"]).astype(int)
    df["gb"] = (df["bb_type"] == "ground_ball").astype(int)
    df["bip"] = df["bb_type"].notna().astype(int)
    return df


def pitcher_splits_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    """(pitcher_id, stand) -> pa, hr, fb, gb, bip."""
    df = _pa_flags(ev, ["pitcher", "stand"])
    return df.groupby(["pitcher", "stand"]).agg(
        pa=("events", "size"), hr=("hr", "sum"), fb=("fb", "sum"),
        gb=("gb", "sum"), bip=("bip", "sum"),
    )


def batter_hrfb_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Per-batter HR/FB (+ fly-ball sample) — the FanGraphs-free fallback."""
    df = _pa_flags(ev, ["batter"])
    g = df.groupby("batter").agg(hr=("hr", "sum"), fb=("fb", "sum"))
    g["hr_fb"] = g["hr"] / g["fb"].replace(0, np.nan)
    return g


def league_rates_from_events(ev: pd.DataFrame) -> dict:
    """League HR/PA, HR/FB, FB rate straight from the season's events."""
    df = _pa_flags(ev, [])
    return {
        "hr_pa": float(df["hr"].sum() / len(df)),
        "hr_fb": float(df["hr"].sum() / max(1, df["fb"].sum())),
        "fb_rate": float(df["fb"].sum() / max(1, df["bip"].sum())),
    }


# ---------------------------------------------------------- strikeouts

_K_EVENTS = ("strikeout", "strikeout_double_play")


def _k_pa(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per PA with a strikeout flag."""
    pa_end = ev["events"].notna() & (ev["events"] != "")
    df = ev.loc[pa_end].copy()
    df["k"] = df["events"].isin(_K_EVENTS).astype(int)
    return df


def league_k_rate_from_events(ev: pd.DataFrame) -> float:
    df = _k_pa(ev)
    return float(df["k"].sum() / max(1, len(df)))


def batter_k_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Per-batter strikeout rate (K per PA) + sample."""
    df = _k_pa(ev)
    g = df.groupby("batter").agg(pa=("k", "size"), k=("k", "sum"))
    g["k_pa"] = g["k"] / g["pa"]
    return g


def pitcher_k_stats_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher K%, plus batters-faced-per-start for starters.

    A game's starter is the pitcher of that half-inning's first PA (same
    detection used for bullpen splits); TBF/start uses only the games a pitcher
    started, so relief cameos don't drag the workload estimate down.
    """
    df = _k_pa(ev)
    g = df.groupby("pitcher").agg(pa=("k", "size"), k=("k", "sum"))
    g["k_pa"] = g["k"] / g["pa"]

    starters = df.loc[
        df.groupby(["game_pk", "inning_topbot"])["at_bat_number"].idxmin(),
        ["game_pk", "inning_topbot", "pitcher"],
    ].rename(columns={"pitcher": "starter"})
    df = df.merge(starters, on=["game_pk", "inning_topbot"], how="left")
    started = df[df["pitcher"] == df["starter"]]
    st = started.groupby("pitcher").agg(
        starter_pa=("k", "size"), starts=("game_pk", "nunique"))
    g = g.join(st)
    g["tbf_per_start"] = g["starter_pa"] / g["starts"]
    return g


def pitcher_overall_stats(season: int) -> pd.DataFrame:
    """FanGraphs fallback (no handedness split): HR/FB and FB% per pitcher."""
    from pybaseball import pitching_stats, playerid_reverse_lookup

    fg = pitching_stats(season, qual=10)[["IDfg", "Name", "TBF", "HR/FB", "FB%"]]
    fg["hr_fb"] = _as_fraction(fg["HR/FB"])
    fg["fb_pct"] = _as_fraction(fg["FB%"])
    ids = playerid_reverse_lookup(fg["IDfg"].tolist(), key_type="fangraphs")
    fg = fg.merge(
        ids[["key_fangraphs", "key_mlbam"]],
        left_on="IDfg", right_on="key_fangraphs", how="inner",
    ).rename(columns={"key_mlbam": "player_id"})
    return fg[["player_id", "TBF", "hr_fb", "fb_pct"]].set_index("player_id")


# --------------------------------------------------------------------- league

def league_baselines(batters: pd.DataFrame, season: int) -> dict:
    """League-average rates the factor ratios are measured against.

    brl/xiso/hr_fb baselines are sample-weighted means of the batter table
    itself (self-consistent with the ratios we compute from it); HR/PA comes
    from FanGraphs team totals with a config fallback.
    """
    from .. import config

    def wmean(rate: str, n: str, fallback: float) -> float:
        d = batters[[rate, n]].dropna()
        if d.empty or d[n].sum() <= 0:
            return fallback
        return float(np.average(d[rate], weights=d[n]))

    base = {
        "brl_pa": wmean("brl_pa", "brl_n", config.LEAGUE_BRL_PA_FALLBACK),
        "xiso": wmean("xiso", "xiso_n", config.LEAGUE_XISO_FALLBACK),
        "hr_fb": wmean("hr_fb", "hr_fb_n", config.LEAGUE_HR_FB_FALLBACK),
        "fb_rate": config.LEAGUE_FB_RATE_FALLBACK,
        "hr_pa": config.LEAGUE_HR_PA_FALLBACK,
    }
    try:
        from pybaseball import team_batting

        tb = team_batting(season)
        if tb["PA"].sum() > 0:
            base["hr_pa"] = float(tb["HR"].sum() / tb["PA"].sum())
    except Exception:
        log.warning("team_batting failed; using fallback league HR/PA", exc_info=True)
    return base


# ---------------------------------------------------------- game lines

def game_scores_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per game: home/away team and final(-ish) runs, from the max
    post-score columns.  Basis for team offense, park runs factors, and the
    league runs-per-game baseline."""
    return ev.groupby("game_pk").agg(
        home_team=("home_team", "first"), away_team=("away_team", "first"),
        home_runs=("post_home_score", "max"), away_runs=("post_away_score", "max"),
    )


def team_offense_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Per team (Savant abbrev): games played and runs scored per game."""
    g = game_scores_from_events(ev)
    home = g[["home_team", "home_runs"]].rename(columns={"home_team": "team", "home_runs": "runs"})
    away = g[["away_team", "away_runs"]].rename(columns={"away_team": "team", "away_runs": "runs"})
    both = pd.concat([home, away])
    out = both.groupby("team").agg(games=("runs", "size"), rpg=("runs", "mean"))
    return out


def park_runs_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Per home park (Savant abbrev): games and average TOTAL runs per game."""
    g = game_scores_from_events(ev)
    g["total"] = g["home_runs"] + g["away_runs"]
    return g.groupby("home_team").agg(games=("total", "size"), total_rpg=("total", "mean"))


def league_rpg_from_events(ev: pd.DataFrame) -> float:
    """League runs per TEAM per game."""
    g = game_scores_from_events(ev)
    return float((g["home_runs"].sum() + g["away_runs"].sum()) / (2 * len(g)))


def pitcher_fip_inputs_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Per pitcher: PA, K, BB(+HBP), HR — the FIP-style skill inputs."""
    pa_end = ev["events"].notna() & (ev["events"] != "")
    df = ev.loc[pa_end, ["pitcher", "events"]].copy()
    df["k"] = df["events"].isin(_K_EVENTS).astype(int)
    df["bb"] = df["events"].isin(["walk", "hit_by_pitch"]).astype(int)
    df["hr"] = (df["events"] == "home_run").astype(int)
    return df.groupby("pitcher").agg(pa=("k", "size"), k=("k", "sum"),
                                     bb=("bb", "sum"), hr=("hr", "sum"))


def bullpen_fip_by_team_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Per pitching team (Savant abbrev): relief-only PA, K, BB(+HBP), HR.
    Starter = pitcher of each half-inning's first PA, as elsewhere."""
    pa_end = ev["events"].notna() & (ev["events"] != "")
    df = ev.loc[pa_end].copy()
    df["pitch_abbrev"] = np.where(df["inning_topbot"] == "Top",
                                  df["home_team"], df["away_team"])
    starters = df.loc[
        df.groupby(["game_pk", "inning_topbot"])["at_bat_number"].idxmin(),
        ["game_pk", "inning_topbot", "pitcher"],
    ].rename(columns={"pitcher": "starter"})
    df = df.merge(starters, on=["game_pk", "inning_topbot"], how="left")
    pen = df[df["pitcher"] != df["starter"]].copy()
    pen["k"] = pen["events"].isin(_K_EVENTS).astype(int)
    pen["bb"] = pen["events"].isin(["walk", "hit_by_pitch"]).astype(int)
    pen["hr"] = (pen["events"] == "home_run").astype(int)
    return pen.groupby("pitch_abbrev").agg(pa=("k", "size"), k=("k", "sum"),
                                           bb=("bb", "sum"), hr=("hr", "sum"))
