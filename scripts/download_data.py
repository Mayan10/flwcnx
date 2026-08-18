#!/usr/bin/env python3
"""Fetch what can be fetched and say plainly what cannot.

The StarNet traces are not in their repo. They sit behind three OneDrive links,
one per country, with no direct download URL, so this script clones the code,
prints the links, and stops. It does not pretend to have downloaded anything.

Also runs the schema inspection, which is the thing to do first on any new drop
of data including the one supplied with the problem statement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

STARNET_REPO = "https://github.com/ConnectedSystemsLab/StarNet"
HORIZON_REPO = "https://github.com/spear-lab/Horizon-Predicting-Starlink-Performance"
HORIZON_DOI = "https://doi.org/10.4121/0bf59468-e5cb-433f-aeb2-e04cf694b65c"

ONEDRIVE_LINKS = {
    "usa": "https://uillinoisedu-my.sharepoint.com/:f:/g/personal/zikunliu_illinois_edu/"
           "IgBaSUJKNIjlQI6KeF1tWc6-AZMwiOlyueqSY4KzOYsVEfw",
    "germany": "https://uillinoisedu-my.sharepoint.com/:f:/g/personal/zikunliu_illinois_edu/"
               "IgBfbwMBpJDTSqbqvty7kK52AalnjRwNyiW-Xomy-iUD01A",
    "canada": "https://uillinoisedu-my.sharepoint.com/:f:/g/personal/zikunliu_illinois_edu/"
              "IgDD3A4WdkPjQ4ME-CCIpk0GAfO40d2gSYm2k5skQMJVf-E",
}


def clone(repo: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {repo} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", repo, str(dest)], check=True)


def starnet(dest: Path) -> int:
    print(f"StarNet code -> {dest / 'code'}")
    clone(STARNET_REPO, dest / "code")

    print("\nThe traces are NOT in the repo. Download each location by hand:\n")
    for location, link in ONEDRIVE_LINKS.items():
        print(f"  {location:8s} {link}")

    print("\nThen place the cleaned pickles at:\n")
    for location in ONEDRIVE_LINKS:
        print(f"  {dest / location / 'dataset_tp_sat.pkl'}")

    print("\nAfterwards, verify the loader against the published statistics:\n")
    print(f"  python scripts/download_data.py --inspect {dest}/usa")
    print("  python -c \"from flwcnx.ingest import ReplaySource; "
          "import json; print(json.dumps(ReplaySource(location='usa')"
          ".verify_against_published(), indent=2))\"")
    return 0


def horizon(dest: Path) -> int:
    print(f"Horizon code -> {dest / 'code'}")
    clone(HORIZON_REPO, dest / "code")
    print(f"\nDataset DOI (BigQuery, not a file download): {HORIZON_DOI}")
    print("Horizon is hourly aggregated with no terminal telemetry, so it")
    print("supports the O1 cross location analysis only, not the calibration.")
    return 0


def inspect(path: Path) -> int:
    """Report how a directory of traces would be mapped. Run this first."""
    from flwcnx.ingest.replay import inspect_schema

    candidates: list[Path] = []
    if path.is_file():
        candidates = [path]
    else:
        for pattern in ("*.pkl", "*.pickle", "*.parquet", "*.csv", "*.csv.gz"):
            candidates += sorted(path.rglob(pattern))
    if not candidates:
        print(f"no trace files under {path}", file=sys.stderr)
        return 1

    for candidate in candidates[:5]:
        print(json.dumps(inspect_schema(candidate), indent=2, default=str))
        print()
    if len(candidates) > 5:
        print(f"... and {len(candidates) - 5} more files")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["starnet", "horizon"], default="starnet")
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--inspect", type=Path, default=None,
                        help="report the column mapping for a file or directory and exit")
    args = parser.parse_args(argv)

    if args.inspect is not None:
        return inspect(args.inspect)
    dest = args.dest or Path("data") / args.dataset
    return starnet(dest) if args.dataset == "starnet" else horizon(dest)


if __name__ == "__main__":
    raise SystemExit(main())
