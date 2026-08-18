#!/usr/bin/env python3
"""Pull historical orbital elements covering the WetLinks measurement window.

One query per day, thinned to the latest element set per satellite, cached
gzipped. Roughly 4,700 satellites and 300 KB per day, so the full window is
about 55 MB and 15 minutes, paid once.

Daily rather than weekly because SGP4 error grows with propagation age, and the
whole point of this is a candidate count that is trustworthy enough to define a
regime with. A stale element set quietly biases the count.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

from flwcnx.ingest.spacetrack import SpaceTrackClient, SpaceTrackError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-09-14")
    parser.add_argument("--end", default="2024-03-12")
    parser.add_argument("--object-name", default="STARLINK")
    args = parser.parse_args(argv)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    client = SpaceTrackClient()

    total_days = (end - start).days + 1
    print(f"fetching {total_days} days, {args.start} to {args.end}", flush=True)

    cursor, done, satellites = start, 0, 0
    while cursor <= end:
        key = f"gp_{args.object_name}_{cursor:%Y%m%d}_{cursor:%Y%m%d}"
        path = (
            f"class/gp_history/EPOCH/{cursor:%Y-%m-%d}--{cursor + timedelta(days=1):%Y-%m-%d}"
            f"/OBJECT_NAME/~~{args.object_name}/orderby/NORAD_CAT_ID/format/json"
            "/predicates/NORAD_CAT_ID,OBJECT_NAME,EPOCH,TLE_LINE1,TLE_LINE2"
        )
        try:
            rows = client.query(path, key)
        except SpaceTrackError as exc:
            print(f"  {cursor}: FAILED {exc}", flush=True)
            return 1
        done += 1
        satellites = len(rows)
        if done % 10 == 0 or done == 1:
            print(f"  {cursor}  {satellites} satellites  ({done}/{total_days})", flush=True)
        cursor += timedelta(days=1)

    print(f"ELEMENTS DONE: {done} days cached", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
