"""Static stadium reference table, keyed by MLB StatsAPI home-team id.

Fields
------
lat / lon        : for the Open-Meteo weather query.
azimuth_deg      : compass bearing (deg clockwise from true north) of the
                   line home plate -> center field.  Used to project the wind
                   vector onto the "out to center" axis.  Values are
                   approximate (+/- ~10 deg) — tune freely.
roof             : 'open' | 'retractable' | 'dome'.  Domes get neutral
                   weather; retractable roofs get a damped weather effect.
hr_pf_rhb/lhb    : HR park factor vs right/left-handed *batters*,
                   100 = neutral.  Hand-built from published Statcast
                   multi-year HR park factors; override via
                   data_cache/park_factors_override.csv (team_id,rhb,lhb)
                   if you want to feed fresher numbers.
"""

from __future__ import annotations

import csv
import os

STADIUMS: dict[int, dict] = {
    108: dict(team="LAA", name="Angel Stadium",            lat=33.800, lon=-117.883, azimuth_deg=65,  roof="open",        hr_pf_rhb=104, hr_pf_lhb=102),
    109: dict(team="AZ",  name="Chase Field",              lat=33.445, lon=-112.067, azimuth_deg=25,  roof="retractable", hr_pf_rhb=103, hr_pf_lhb=106),
    110: dict(team="BAL", name="Oriole Park at Camden Yards", lat=39.284, lon=-76.622, azimuth_deg=30, roof="open",       hr_pf_rhb=96,  hr_pf_lhb=98),
    111: dict(team="BOS", name="Fenway Park",              lat=42.346, lon=-71.097,  azimuth_deg=52,  roof="open",        hr_pf_rhb=96,  hr_pf_lhb=88),
    112: dict(team="CHC", name="Wrigley Field",            lat=41.948, lon=-87.656,  azimuth_deg=35,  roof="open",        hr_pf_rhb=102, hr_pf_lhb=100),
    113: dict(team="CIN", name="Great American Ball Park", lat=39.097, lon=-84.507,  azimuth_deg=120, roof="open",        hr_pf_rhb=120, hr_pf_lhb=118),
    114: dict(team="CLE", name="Progressive Field",        lat=41.496, lon=-81.685,  azimuth_deg=0,   roof="open",        hr_pf_rhb=98,  hr_pf_lhb=102),
    115: dict(team="COL", name="Coors Field",              lat=39.756, lon=-104.994, azimuth_deg=5,   roof="open",        hr_pf_rhb=113, hr_pf_lhb=115),
    116: dict(team="DET", name="Comerica Park",            lat=42.339, lon=-83.049,  azimuth_deg=145, roof="open",        hr_pf_rhb=94,  hr_pf_lhb=96),
    117: dict(team="HOU", name="Daikin Park",              lat=29.757, lon=-95.356,  azimuth_deg=345, roof="retractable", hr_pf_rhb=110, hr_pf_lhb=96),
    118: dict(team="KC",  name="Kauffman Stadium",         lat=39.051, lon=-94.480,  azimuth_deg=45,  roof="open",        hr_pf_rhb=88,  hr_pf_lhb=90),
    119: dict(team="LAD", name="Dodger Stadium",           lat=34.074, lon=-118.240, azimuth_deg=25,  roof="open",        hr_pf_rhb=110, hr_pf_lhb=112),
    120: dict(team="WSH", name="Nationals Park",           lat=38.873, lon=-77.007,  azimuth_deg=30,  roof="open",        hr_pf_rhb=100, hr_pf_lhb=102),
    121: dict(team="NYM", name="Citi Field",               lat=40.757, lon=-73.846,  azimuth_deg=15,  roof="open",        hr_pf_rhb=103, hr_pf_lhb=98),
    133: dict(team="ATH", name="Sutter Health Park",       lat=38.580, lon=-121.513, azimuth_deg=60,  roof="open",        hr_pf_rhb=103, hr_pf_lhb=103),
    134: dict(team="PIT", name="PNC Park",                 lat=40.447, lon=-80.006,  azimuth_deg=120, roof="open",        hr_pf_rhb=88,  hr_pf_lhb=98),
    135: dict(team="SD",  name="Petco Park",               lat=32.707, lon=-117.157, azimuth_deg=0,   roof="open",        hr_pf_rhb=95,  hr_pf_lhb=93),
    136: dict(team="SEA", name="T-Mobile Park",            lat=47.591, lon=-122.332, azimuth_deg=45,  roof="retractable", hr_pf_rhb=96,  hr_pf_lhb=94),
    137: dict(team="SF",  name="Oracle Park",              lat=37.778, lon=-122.389, azimuth_deg=85,  roof="open",        hr_pf_rhb=85,  hr_pf_lhb=78),
    138: dict(team="STL", name="Busch Stadium",            lat=38.623, lon=-90.193,  azimuth_deg=65,  roof="open",        hr_pf_rhb=92,  hr_pf_lhb=90),
    139: dict(team="TB",  name="Tropicana Field",          lat=27.768, lon=-82.653,  azimuth_deg=45,  roof="dome",        hr_pf_rhb=95,  hr_pf_lhb=97),
    140: dict(team="TEX", name="Globe Life Field",         lat=32.747, lon=-97.084,  azimuth_deg=45,  roof="retractable", hr_pf_rhb=98,  hr_pf_lhb=102),
    141: dict(team="TOR", name="Rogers Centre",            lat=43.641, lon=-79.389,  azimuth_deg=15,  roof="retractable", hr_pf_rhb=105, hr_pf_lhb=108),
    142: dict(team="MIN", name="Target Field",             lat=44.982, lon=-93.278,  azimuth_deg=90,  roof="open",        hr_pf_rhb=98,  hr_pf_lhb=95),
    143: dict(team="PHI", name="Citizens Bank Park",       lat=39.906, lon=-75.166,  azimuth_deg=10,  roof="open",        hr_pf_rhb=112, hr_pf_lhb=110),
    144: dict(team="ATL", name="Truist Park",              lat=33.891, lon=-84.468,  azimuth_deg=30,  roof="open",        hr_pf_rhb=105, hr_pf_lhb=107),
    145: dict(team="CWS", name="Rate Field",               lat=41.830, lon=-87.634,  azimuth_deg=35,  roof="open",        hr_pf_rhb=110, hr_pf_lhb=108),
    146: dict(team="MIA", name="loanDepot park",           lat=25.778, lon=-80.220,  azimuth_deg=40,  roof="retractable", hr_pf_rhb=92,  hr_pf_lhb=90),
    147: dict(team="NYY", name="Yankee Stadium",           lat=40.829, lon=-73.926,  azimuth_deg=75,  roof="open",        hr_pf_rhb=108, hr_pf_lhb=118),
    158: dict(team="MIL", name="American Family Field",    lat=43.028, lon=-87.971,  azimuth_deg=135, roof="retractable", hr_pf_rhb=110, hr_pf_lhb=108),
}

_OVERRIDE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_cache", "park_factors_override.csv",
)


def _apply_overrides() -> None:
    """Optionally override park factors from a CSV: team_id,hr_pf_rhb,hr_pf_lhb."""
    if not os.path.exists(_OVERRIDE_PATH):
        return
    with open(_OVERRIDE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            tid = int(row["team_id"])
            if tid in STADIUMS:
                STADIUMS[tid]["hr_pf_rhb"] = float(row["hr_pf_rhb"])
                STADIUMS[tid]["hr_pf_lhb"] = float(row["hr_pf_lhb"])


_apply_overrides()


def stadium_for_home_team(team_id: int) -> dict | None:
    return STADIUMS.get(team_id)
