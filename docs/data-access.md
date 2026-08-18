# Getting the StarNet traces

Status as of 2026-08-18: **blocked**. All three OneDrive links in the repo
README have expired. Everything below was checked, not assumed.

## What was ruled out

| Route | Result |
|---|---|
| Repo README links (US, Germany, Canada) | expired, all three |
| Repo issues / forks / mirrors | none exist |
| ACM DL entry (10.1145/3768971) | no artifact DOI, no supplementary data |
| `ConnectedSystemsLab` other repos | `starloc_preproc`, `SpaceGEN`, `StarCDN-Simulator` are different projects |
| Zenodo / Kaggle re-hosting | nothing found for these traces |

## Option A: ask the authors (recommended)

The only route that returns the actual dataset. Draft is in
`docs/email-draft.md`. Send from Mayan's own account.

Likely addresses, from the OneDrive personal path `zikunliu_illinois_edu` and
UIUC's convention:

- Zikun Liu, `zikunliu2@illinois.edu` (first author, owner of the share)
- Deepak Vasisht, `deepakv@illinois.edu` (PI, most likely to reply)

Expect days, not hours. Start this before anything else because it is the only
option whose clock is not ours to control.

## Option B: LEOScope

https://leoscope.surrey.ac.uk

A global LEO testbed, open to non-commercial research, that gathers exactly the
telemetry this project needs. The StarNet paper used it for part of their
collection. This would mean *collecting* rather than downloading, so it is
heavier and slower, but it is the only route that produces a dataset we control
and could extend.

Worth applying to in parallel even if Option A succeeds, if the timeline
allows.

## Option C: degraded path on a public 1 Hz trace

No public dataset has all four of what the regime layer needs (1 Hz throughput,
serving satellite identity, its elevation and distance, candidate count) from a
**stationary** terminal. What exists:

| Dataset | 1 Hz throughput | Satellite geometry | Note |
|---|---|---|---|
| LENS (already cited, entry 11) | no, latency only | no | has obstruction maps, 20 locations, Zenodo |
| WetLinks (entry 9) | no, ~1 measurement per 3-4 min | no | 140k measurements, 2 sites, professional weather |
| Starlink Robot (arXiv 2506.19781) | yes | yes, per visible satellite | **mobile robot**, different regime, access unclear |
| Horizon (entry 14) | no, hourly aggregate | no | already scoped to O1 only |

The useful observation is that **two of the four regime axes are recoverable
from any 1 Hz trace with a timestamp and a location**, using the SGP4 code
already in `ingest/tle.py`:

- `phase` needs only timestamps, and `state/phase.py` recovers it;
- `candidate_count` is `tle.candidate_count(observer, tles, when)`, exact.

What is not recoverable without obstruction maps is *which* satellite is
serving, and therefore its elevation and distance. So a degraded run would
condition on phase and candidates only. That is not a crippling loss: the
granularity ablation in `eval/runner.py` exists precisely to ask whether phase
alone carries the gain, and a degraded run answers a weaker version of the same
question. It would have to be reported as such, not as the full result.

## Option D: the supplied dataset

Still unseen. If the dataset handed over with the problem statement has 1 Hz
throughput and any satellite telemetry, it may make all of the above moot and
becomes the primary source. Inspect first:

```
python scripts/download_data.py --inspect data/supplied
```
