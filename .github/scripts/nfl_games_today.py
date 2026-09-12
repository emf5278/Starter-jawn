"""Print "<games today> <window>" for a date (default: today, ET).

Standard library only, on purpose: the touchdown workflow's gate step runs
this *before* setting up Python or installing requirements, so that a day
with no NFL games costs a couple of seconds and zero Odds API credits.

The window is the same early/late split the pipeline uses -- resolved here
too so the gate can tell an already-published early board apart from a late
one that still needs to run. It mirrors pipeline.data.nfl.resolve_window:
before 1pm ET it is "early", after that "late", and either falls back to
"all" when the day has no games on that side of the 4pm ET split (a lone
Thursday or Monday night game must not vanish from a morning run).

Prints "1 all" on a fetch failure rather than "0", so a transient network
problem makes the workflow run (and exit cleanly on its own) instead of
silently skipping a real game day.
"""
import csv
import datetime as dt
import io
import sys
import urllib.request
import zoneinfo

URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
ET = zoneinfo.ZoneInfo("America/New_York")

SPLIT_ET_HOUR = 16          # keep in step with config.TD_WINDOW_SPLIT_ET_HOUR
AUTO_SWITCH_ET_HOUR = 13    # keep in step with config.TD_WINDOW_AUTO_SWITCH_ET_HOUR


def _kickoff_hour(gametime: str) -> int | None:
    if not gametime or ":" not in gametime:
        return None
    try:
        return int(gametime.split(":")[0])
    except ValueError:
        return None


def main() -> None:
    now = dt.datetime.now(ET)
    day = sys.argv[1] if len(sys.argv) > 1 else now.date().isoformat()
    try:
        raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8", "replace")
    except Exception as e:                       # noqa: BLE001
        print(f"schedule fetch failed: {e}", file=sys.stderr)
        print("1 all")
        return

    hours = [_kickoff_hour(r.get("gametime", ""))
             for r in csv.DictReader(io.StringIO(raw)) if r.get("gameday") == day]
    n = len(hours)
    window = "early" if now.hour < AUTO_SWITCH_ET_HOUR else "late"
    in_window = [h for h in hours
                 if h is None
                 or (window == "early" and h < SPLIT_ET_HOUR)
                 or (window == "late" and h >= SPLIT_ET_HOUR)]
    if not in_window:
        window = "all"
    print(f"{n} {window}")


if __name__ == "__main__":
    main()
