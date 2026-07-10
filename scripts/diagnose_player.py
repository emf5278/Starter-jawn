"""One-off diagnostic: why is a given player where they are in the rankings?

    python -m scripts.diagnose_player "Caminero" [--date YYYY-MM-DD]

Reproduces the daily scoring for the whole slate (odds/BvP/bullpen left neutral
for speed — they don't affect the probability ranking materially) and prints,
for every player whose name matches:

  * VENUE CHECK — the venue MLB StatsAPI reports for the game vs. the park the
    model assumes from pipeline/stadiums.py (name, roof, HR park factor).  A
    mismatch means stale park/weather inputs.
  * LINEUP — found in the lineup? which slot? confirmed or projected?
  * STATS — is the player's id present in the Statcast/FanGraphs join? the
    barrels/PA, HR/FB, xISO the model used (a batter factor of ~1.00 means the
    stats did not join and the hitter was scored as league-average).
  * FACTORS + PROB — the full breakdown and where the player ranks by model
    probability among everyone scored today.

Not part of the pipeline; safe to delete.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from zoneinfo import ZoneInfo

from pipeline import config
from pipeline.data import lineups, statcast
from pipeline.model.predict import predict_player
from pipeline.stadiums import stadium_for_home_team


def _batter_hand(bat_side: str, pitcher_hand: str | None) -> str:
    if bat_side == "S":
        return "L" if (pitcher_hand or "R") == "R" else "R"
    return bat_side or "R"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="player name substring, e.g. Caminero")
    ap.add_argument("--date")
    args = ap.parse_args()
    date = (dt.date.fromisoformat(args.date) if args.date
            else dt.datetime.now(ZoneInfo("America/New_York")).date())
    q = args.query.lower()
    season = date.year

    print(f"=== diagnose '{args.query}' for {date} ===\n")
    slate = lineups.todays_slate(date)
    players = slate["players"]
    batters = statcast.batter_power_stats(season)
    league = statcast.league_baselines(batters, season)
    events = statcast.season_events(season, date)
    splits = None
    if events is not None:
        splits = statcast.pitcher_splits_from_events(events)
        league.update(statcast.league_rates_from_events(events))
        hrfb = statcast.batter_hrfb_from_events(events)
        batters = batters.join(
            hrfb.rename(columns={"hr_fb": "hr_fb_ev", "fb": "fb_ev"}), how="left")
        batters["hr_fb"] = batters["hr_fb"].fillna(batters["hr_fb_ev"])
        batters["hr_fb_n"] = (
            batters["hr_fb_n"].mask(batters["hr_fb_n"] <= 0).fillna(batters["fb_ev"]))
    print("league baselines:", {k: round(v, 4) for k, v in league.items()}, "\n")

    all_rows, hits = [], []
    for game in slate["games"]:
        home_id = game["teams"]["home"]["team_id"]
        stadium = stadium_for_home_team(home_id)
        statsapi_venue = game.get("venue")
        wx = None
        if stadium is not None:
            from pipeline.data import weather as weather_mod
            wx = weather_mod.game_weather(
                stadium["lat"], stadium["lon"], game["game_time_utc"], stadium["azimuth_deg"])
        for side, opp in (("home", "away"), ("away", "home")):
            team, opp_team = game["teams"][side], game["teams"][opp]
            pid_p = opp_team["probable_pitcher_id"]
            phand = players.get(pid_p, {}).get("pitch_hand") if pid_p else None
            for slot, pid in enumerate(team["lineup"], start=1):
                info = players.get(pid, {})
                hand = _batter_hand(info.get("bat_side", "R"), phand)
                found = pid in batters.index
                stats = batters.loc[pid].to_dict() if found else {}
                split = None
                if splits is not None and pid_p is not None:
                    try:
                        r = splits.loc[(pid_p, hand)]
                        split = {k: float(r[k]) for k in ("pa", "hr", "fb", "gb", "bip")}
                    except KeyError:
                        pass
                pred = predict_player(stats, split, stadium, wx, hand, slot, league) \
                    if stadium else {"prob": 0}
                rec = dict(pid=pid, name=info.get("name", str(pid)), team=team["team_name"],
                           opp=opp_team["team_name"], slot=slot,
                           confirmed=team["lineup_confirmed"], hand=hand, found=found,
                           statsapi_venue=statsapi_venue, stadium=stadium, stats=stats,
                           pitcher=opp_team["probable_pitcher_name"], pred=pred)
                all_rows.append(rec)
                if q in rec["name"].lower():
                    hits.append(rec)

    all_rows.sort(key=lambda r: r["pred"]["prob"], reverse=True)
    rank = {id(r): i + 1 for i, r in enumerate(all_rows)}
    n = len(all_rows)

    if not hits:
        print(f"NO MATCH for '{args.query}' in any lineup today.")
        print("-> the player is not in a fetched lineup (rest day, not posted, "
              "or fallback lineup missed him).")
        return

    for r in hits:
        s, st, pr = r["stats"], r["stadium"], r["pred"]
        f = pr.get("factors", {})
        print(f"### {r['name']}  (MLBAM {r['pid']}, bats {r['hand']})")
        print(f"  team {r['team']} vs {r['opp']}, pitcher {r['pitcher']}")
        print(f"  lineup slot {r['slot']} ({'confirmed' if r['confirmed'] else 'PROJECTED'})")
        print(f"  RANK by model prob: #{rank[id(r)]} of {n}  (prob {pr['prob']})")
        print("  VENUE CHECK:")
        print(f"    StatsAPI says game venue = {r['statsapi_venue']!r}")
        if st:
            print(f"    model assumes           = {st['name']!r} "
                  f"roof={st['roof']} HRpf(RHB/LHB)={st['hr_pf_rhb']}/{st['hr_pf_lhb']}")
            if r["statsapi_venue"] and st["name"] not in r["statsapi_venue"] \
                    and r["statsapi_venue"] not in st["name"]:
                print("    *** MISMATCH — model park/weather inputs are STALE ***")
        print(f"  STATS FOUND IN JOIN: {r['found']}")
        if r["found"]:
            print(f"    brl_pa={s.get('brl_pa')} (n={s.get('brl_n')})  "
                  f"hr_fb={s.get('hr_fb')} (n={s.get('hr_fb_n')})  "
                  f"xiso={s.get('xiso')} (n={s.get('xiso_n')})")
        else:
            print("    *** player id NOT in stats table -> scored as league-average ***")
        if f:
            print(f"    batter factor ={f['batter']['value']}  park={f['park']['value']}  "
                  f"weather={f['weather']['value']}  starter={f['pitcher']['value']}")
            print(f"    expected_pa={pr['expected_pa']}  per-PA vs starter={pr['per_pa_prob_vs_starter']}")
        print()

    cut = all_rows[min(19, n - 1)]["pred"]["prob"]
    print(f"top-20 probability cutoff today: {cut}")


if __name__ == "__main__":
    sys.exit(main())
