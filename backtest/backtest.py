"""Backtest the HR model over a past season.

    python backtest/backtest.py --season 2024 [--start 2024-05-01] [--end 2024-09-29]
    python backtest/backtest.py --season 2024 --odds --odds-days 14

What it does
------------
1. Downloads the season's raw Statcast events (pybaseball, cached on disk).
2. Rebuilds, per player per day, the cumulative inputs the live model uses
   (barrels/PA, HR/FB, xISO proxy from estimated stats; pitcher handedness
   splits) — joined with merge_asof(allow_exact_matches=False) so each
   prediction sees only data through the PREVIOUS day. Lineup slots and
   starters are inferred from the event stream itself.
3. Scores P(>=1 HR) for every starter-vs-lineup matchup and compares to what
   actually happened.

Reports
-------
* Reliability (calibration) curve -> backtest/output/calibration.png + .json
* Brier score, log loss, base rate vs mean prediction
* Daily top-10 hit rate (the dashboard's headline list)
* With --odds (requires ODDS_API_KEY on a plan with historical access):
  flat-stake ROI of +EV picks vs the 13:00 UTC snapshot, and CLV vs the
  closing snapshot. Historical odds cost API credits — bounded by --odds-days.

Known simplifications (documented, deliberate):
* Weather factor is not reconstructed historically (factor = 1.0).
* League baselines use the full season (negligible leakage).
* Switch hitters use the batter side actually recorded for that game's
  first PA — equivalent to knowing the starter's hand in advance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import config  # noqa: E402
from pipeline.model.factors import batter_power_factor, park_factor, pitcher_hr_factor  # noqa: E402
from pipeline.model.predict import ev_per_dollar  # noqa: E402
from pipeline.stadiums import STADIUMS  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Savant home_team abbreviation -> stadium record (plus parks that no longer
# host: needed when backtesting seasons before a move).
PARKS_BY_ABBREV = {s["team"]: s for s in STADIUMS.values()}
PARKS_BY_ABBREV.update({
    "OAK": dict(team="OAK", name="Oakland Coliseum", lat=37.752, lon=-122.201,
                azimuth_deg=55, roof="open", hr_pf_rhb=91, hr_pf_lhb=93),
    "ARI": PARKS_BY_ABBREV.get("AZ"),
    "CHW": PARKS_BY_ABBREV.get("CWS"),
    "WSN": PARKS_BY_ABBREV.get("WSH"),
    "TBR": PARKS_BY_ABBREV.get("TB"),
    "KCR": PARKS_BY_ABBREV.get("KC"),
    "SDP": PARKS_BY_ABBREV.get("SD"),
    "SFG": PARKS_BY_ABBREV.get("SF"),
})


def is_barrel(ev: pd.Series, la: pd.Series) -> pd.Series:
    """Statcast barrel approximation: >=98 mph EV, launch-angle window that
    widens with EV (26-30 deg at 98, -1/+2 deg per extra mph, clamped 8-50)."""
    lo = np.maximum(8.0, 26.0 - (ev - 98.0))
    hi = np.minimum(50.0, 30.0 + 2.0 * (ev - 98.0))
    return (ev >= 98.0) & (la >= lo) & (la <= hi)


def load_events(season: int, start: str, end: str) -> pd.DataFrame:
    from pybaseball import cache, statcast
    cache.enable()
    print(f"downloading statcast {season} events (cached after first run)...")
    ev = statcast(start_dt=f"{season}-03-20", end_dt=end, verbose=False)
    ev = ev[ev["game_type"] == "R"].copy()
    ev["game_date"] = pd.to_datetime(ev["game_date"])
    return ev


def pa_events(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per plate appearance, with the derived flags the model needs."""
    pa = ev[ev["events"].notna() & (ev["events"] != "")].copy()
    pa["hr"] = (pa["events"] == "home_run").astype(int)
    pa["so"] = pa["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    pa["bip"] = pa["bb_type"].notna().astype(int)
    pa["fb"] = pa["bb_type"].isin(["fly_ball", "popup"]).astype(int)
    pa["gb"] = (pa["bb_type"] == "ground_ball").astype(int)
    pa["barrel"] = is_barrel(
        pd.to_numeric(pa["launch_speed"], errors="coerce"),
        pd.to_numeric(pa["launch_angle"], errors="coerce"),
    ).fillna(False).astype(int)
    for col in ("estimated_slg_using_speedangle", "estimated_ba_using_speedangle"):
        pa[col] = pd.to_numeric(pa.get(col), errors="coerce").fillna(0.0) * pa["bip"]
    pa["ab"] = pa["so"] + pa["bip"]  # ISO-style denominator
    return pa


def cumulative_batter_stats(pa: pd.DataFrame) -> pd.DataFrame:
    """Per (batter, date): season-to-date totals BEFORE that date's games."""
    day = pa.groupby(["batter", "game_date"]).agg(
        pa_n=("hr", "size"), hr=("hr", "sum"), fb=("fb", "sum"), ab=("ab", "sum"),
        barrels=("barrel", "sum"),
        xslg=("estimated_slg_using_speedangle", "sum"),
        xba=("estimated_ba_using_speedangle", "sum"),
    ).reset_index().sort_values(["batter", "game_date"])
    cum = day.groupby("batter")[["pa_n", "hr", "fb", "ab", "barrels", "xslg", "xba"]].cumsum()
    cum.columns = [f"c_{c}" for c in cum.columns]
    out = pd.concat([day[["batter", "game_date"]], cum], axis=1)
    out["brl_pa"] = out["c_barrels"] / out["c_pa_n"]
    out["hr_fb"] = out["c_hr"] / out["c_fb"].replace(0, np.nan)
    out["xiso"] = (out["c_xslg"] - out["c_xba"]) / out["c_ab"].replace(0, np.nan)
    return out[["batter", "game_date", "brl_pa", "c_pa_n", "hr_fb", "c_fb", "xiso", "c_ab"]]


def cumulative_pitcher_splits(pa: pd.DataFrame) -> pd.DataFrame:
    day = pa.groupby(["pitcher", "stand", "game_date"]).agg(
        pa_n=("hr", "size"), hr=("hr", "sum"), fb=("fb", "sum"),
        gb=("gb", "sum"), bip=("bip", "sum"),
    ).reset_index().sort_values(["pitcher", "stand", "game_date"])
    cum = day.groupby(["pitcher", "stand"])[["pa_n", "hr", "fb", "gb", "bip"]].cumsum()
    cum.columns = [f"p_{c}" for c in cum.columns]
    return pd.concat([day[["pitcher", "stand", "game_date"]], cum], axis=1)


def build_matchups(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, batter): inferred slot, stand, opposing starter, park,
    and the outcome (>=1 HR in that game)."""
    pa = pa.sort_values(["game_pk", "at_bat_number"])
    first = pa.groupby(["game_pk", "inning_topbot", "batter"], as_index=False).agg(
        first_ab=("at_bat_number", "min"), stand=("stand", "first"),
        date=("game_date", "first"), home=("home_team", "first"),
    )
    first["slot"] = first.sort_values("first_ab").groupby(
        ["game_pk", "inning_topbot"]).cumcount() + 1
    first = first[first["slot"] <= 9]  # starters only; skip pinch hitters

    # opposing starter = pitcher of each half-inning's first event
    starters = pa.loc[pa.groupby(["game_pk", "inning_topbot"])["at_bat_number"].idxmin(),
                      ["game_pk", "inning_topbot", "pitcher", "p_throws"]]
    m = first.merge(starters, on=["game_pk", "inning_topbot"])

    outcome = pa.groupby(["game_pk", "batter"])["hr"].sum().rename("hr_hit").reset_index()
    m = m.merge(outcome, on=["game_pk", "batter"], how="left")
    m["y"] = (m["hr_hit"] > 0).astype(int)
    return m


def league_from_events(pa: pd.DataFrame) -> dict:
    bip = pa["bip"].sum()
    return {
        "hr_pa": pa["hr"].sum() / len(pa),
        "brl_pa": pa["barrel"].sum() / len(pa),
        "hr_fb": pa["hr"].sum() / max(1, pa["fb"].sum()),
        "fb_rate": pa["fb"].sum() / max(1, bip),
        "xiso": (pa["estimated_slg_using_speedangle"].sum()
                 - pa["estimated_ba_using_speedangle"].sum()) / max(1, pa["ab"].sum()),
    }


def score(m: pd.DataFrame, league: dict) -> pd.DataFrame:
    """Row-wise model probability. Weather = neutral (not reconstructed)."""
    def one(r):
        stats = {"brl_pa": r["brl_pa"], "brl_n": r["c_pa_n"],
                 "hr_fb": r["hr_fb"], "hr_fb_n": r["c_fb"],
                 "xiso": r["xiso"], "xiso_n": r["c_ab"]}
        b = batter_power_factor(stats, league)["value"]
        split = None
        if pd.notna(r.get("p_pa_n")):
            split = {"pa": r["p_pa_n"], "hr": r["p_hr"], "fb": r["p_fb"],
                     "gb": r["p_gb"], "bip": r["p_bip"]}
        p = pitcher_hr_factor(split, league)["value"]
        park = PARKS_BY_ABBREV.get(r["home"])
        k = park_factor(park, r["stand"])["value"] if park else 1.0

        base = league["hr_pa"] * b * k
        p_st = min(config.PER_PA_PROB_CAP, base * p)
        p_bp = min(config.PER_PA_PROB_CAP, base)
        epa = config.expected_pa_for_slot(int(r["slot"]))
        n_st = epa * config.STARTER_PA_SHARE
        return 1 - (1 - p_st) ** n_st * (1 - p_bp) ** (epa - n_st)

    m = m.copy()
    m["prob"] = m.apply(one, axis=1)
    return m


# ------------------------------------------------------------------ reports

def reliability(m: pd.DataFrame, bins: int = 10) -> dict:
    q = pd.qcut(m["prob"], bins, duplicates="drop")
    g = m.groupby(q, observed=True).agg(pred=("prob", "mean"), obs=("y", "mean"),
                                        n=("y", "size")).reset_index(drop=True)
    eps = 1e-12
    p = m["prob"].clip(eps, 1 - eps)
    return {
        "bins": g.to_dict("records"),
        "brier": float(((m["prob"] - m["y"]) ** 2).mean()),
        "log_loss": float(-(m["y"] * np.log(p) + (1 - m["y"]) * np.log(1 - p)).mean()),
        "base_rate": float(m["y"].mean()),
        "mean_prediction": float(m["prob"].mean()),
        "n": int(len(m)),
    }


def plot_reliability(rep: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, blue = "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"
    bins = rep["bins"]
    pred = [b["pred"] for b in bins]
    obs = [b["obs"] for b in bins]
    n = [b["n"] for b in bins]
    lim = max(max(pred), max(obs)) * 1.15

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(6.4, 7), height_ratios=[4, 1], sharex=True,
        facecolor="#fcfcfb", gridspec_kw={"hspace": 0.08})
    for a in (ax, axn):
        a.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(grid)
        a.tick_params(colors=ink2, labelsize=9)
        a.grid(color=grid, linewidth=0.6)
    ax.plot([0, lim], [0, lim], color="#c3c2b7", linewidth=1, linestyle="--",
            label="perfect calibration")
    ax.plot(pred, obs, color=blue, linewidth=2, marker="o", markersize=6,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5, label="model")
    ax.set_ylabel("observed HR rate", color=ink2, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=ink2)
    ax.set_title(
        f"Reliability curve — Brier {rep['brier']:.4f}, log loss {rep['log_loss']:.4f}, "
        f"n={rep['n']:,}\nbase rate {rep['base_rate']:.3f} vs mean prediction "
        f"{rep['mean_prediction']:.3f}", color=ink, fontsize=10, loc="left")
    axn.bar(pred, n, width=lim / 40, color=blue)
    axn.set_ylabel("n", color=ink2, fontsize=9)
    axn.set_xlabel("predicted P(≥1 HR)", color=ink2, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def top10_report(m: pd.DataFrame) -> dict:
    daily = m.sort_values("prob", ascending=False).groupby("date").head(10)
    return {
        "days": int(daily["date"].nunique()),
        "picks": int(len(daily)),
        "hit_rate": float(daily["y"].mean()),
        "mean_prediction": float(daily["prob"].mean()),
    }


# ------------------------------------------------------------- historical odds

def odds_report(m: pd.DataFrame, season: int, days: int) -> dict | None:
    """ROI of flat $1 bets on +EV picks at the morning snapshot, and CLV vs
    close. Uses The Odds API historical endpoints (paid feature; costs credits
    proportional to events x days — bounded by --odds-days)."""
    import requests
    from pybaseball import playerid_reverse_lookup

    from pipeline.data.odds import normalize_name

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("--odds requested but ODDS_API_KEY not set; skipping")
        return None

    ids = m["batter"].unique().tolist()
    lk = playerid_reverse_lookup(ids, key_type="mlbam")
    names = {r.key_mlbam: normalize_name(f"{r.name_first} {r.name_last}")
             for r in lk.itertuples()}
    m = m.copy()
    m["norm_name"] = m["batter"].map(names)

    dates = sorted(m["date"].dt.date.unique())[-days:]
    base = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb"

    def snapshot(ts: str) -> dict[str, float]:
        """normalized player name -> best Over 0.5 decimal price at `ts`."""
        best: dict[str, float] = {}
        try:
            evs = requests.get(f"{base}/events",
                               params={"apiKey": key, "date": ts}, timeout=30)
            evs.raise_for_status()
            for ev in evs.json().get("data", []):
                od = requests.get(
                    f"{base}/events/{ev['id']}/odds",
                    params={"apiKey": key, "date": ts, "regions": "us",
                            "markets": config.ODDS_MARKET, "oddsFormat": "decimal"},
                    timeout=30)
                if od.status_code != 200:
                    continue
                for bk in od.json().get("data", {}).get("bookmakers", []):
                    for mk in bk.get("markets", []):
                        for o in mk.get("outcomes", []):
                            if o.get("name") in ("Over", "Yes"):
                                nm = normalize_name(o.get("description", ""))
                                best[nm] = max(best.get(nm, 0), float(o["price"]))
        except Exception as e:  # noqa: BLE001
            print(f"  historical odds fetch failed at {ts}: {e}")
        return best

    bets = []
    for d in dates:
        entry = snapshot(f"{d}T13:00:00Z")
        close = snapshot(f"{d}T22:55:00Z")  # ~ before most first pitches
        if not entry:
            continue
        day = m[m["date"].dt.date == d]
        for r in day.itertuples():
            price = entry.get(r.norm_name)
            if not price:
                continue
            ev = ev_per_dollar(r.prob, price)
            if ev <= 0:
                continue
            cp = close.get(r.norm_name)
            bets.append({
                "date": str(d), "prob": r.prob, "price": price, "y": r.y,
                "clv_pct": (price / cp - 1) if cp else None,
            })
        print(f"  {d}: {len([b for b in bets if b['date'] == str(d)])} +EV bets")

    if not bets:
        return {"note": "no bets placed (no odds data or no +EV spots)", "n_bets": 0}
    pnl = sum(b["y"] * (b["price"] - 1) - (1 - b["y"]) for b in bets)
    clvs = [b["clv_pct"] for b in bets if b["clv_pct"] is not None]
    return {
        "n_bets": len(bets),
        "roi_flat_stake": pnl / len(bets),
        "hit_rate": sum(b["y"] for b in bets) / len(bets),
        "mean_clv_pct": float(np.mean(clvs)) if clvs else None,
        "days_covered": len(dates),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--start", help="first prediction date (default May 1)")
    ap.add_argument("--end", help="last date (default Sep 29)")
    ap.add_argument("--odds", action="store_true", help="historical ROI/CLV (paid API)")
    ap.add_argument("--odds-days", type=int, default=14)
    args = ap.parse_args()

    start = args.start or f"{args.season}-05-01"
    end = args.end or f"{args.season}-09-29"
    os.makedirs(OUT_DIR, exist_ok=True)

    ev = load_events(args.season, start, end)
    pa = pa_events(ev)
    league = league_from_events(pa)
    print("league baselines:", {k: round(v, 4) for k, v in league.items()})

    bat = cumulative_batter_stats(pa)
    pit = cumulative_pitcher_splits(pa)
    m = build_matchups(pa)
    m = m[(m["date"] >= start) & (m["date"] <= end)]
    print(f"{len(m):,} matchups from {start} to {end}")

    # leak-free join: stats strictly BEFORE each prediction date
    m = pd.merge_asof(m.sort_values("date"), bat.sort_values("game_date"),
                      left_on="date", right_on="game_date", left_by="batter",
                      right_by="batter", allow_exact_matches=False)
    m = pd.merge_asof(m.sort_values("date"), pit.sort_values("game_date"),
                      left_on="date", right_on="game_date",
                      left_by=["pitcher", "stand"], right_by=["pitcher", "stand"],
                      allow_exact_matches=False)

    m = score(m, league)
    rep = reliability(m)
    rep["top10"] = top10_report(m)
    if args.odds:
        rep["odds"] = odds_report(m, args.season, args.odds_days)

    plot_reliability(rep, os.path.join(OUT_DIR, "calibration.png"))
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)

    print(json.dumps({k: v for k, v in rep.items() if k != "bins"}, indent=2, default=str))
    print(f"full report: {OUT_DIR}/report.json")


if __name__ == "__main__":
    main()
