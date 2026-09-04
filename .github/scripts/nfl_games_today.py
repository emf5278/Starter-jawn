"""Print how many NFL games kick off on a given date (default: today, ET).

Standard library only, on purpose: the touchdown workflow's gate step runs
this *before* setting up Python or installing requirements, so that a day
with no NFL games costs a couple of seconds and zero Odds API credits.

Prints 1 on a fetch failure rather than 0, so a transient network problem
makes the workflow run (and exit cleanly on its own) instead of silently
skipping a real game day.
"""
import csv
import datetime as dt
import io
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def main() -> None:
    if len(sys.argv) > 1:
        day = sys.argv[1]
    else:
        day = dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).date().isoformat()
    try:
        raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8", "replace")
    except Exception as e:                       # noqa: BLE001
        print(f"schedule fetch failed: {e}", file=sys.stderr)
        print(1)
        return
    n = sum(1 for r in csv.DictReader(io.StringIO(raw)) if r.get("gameday") == day)
    print(n)


if __name__ == "__main__":
    main()
