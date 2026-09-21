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

---

# Resolution, 2026-08-31

**The traces were never obtained.** The author email went unanswered, and the
OneDrive links are still expired. Option A is closed for now. What follows is
what was done instead, and what it costs.

## The workaround

Option C, taken seriously rather than as a fallback. The key was that the
supplied-dataset inspection had looked at only part of WetLinks. The full
public release at https://github.com/sys-uos/WetLinks contains
`iperf_cleaned_seconds_*.csv`, and that file is what makes the project
runnable:

| what the project needs | StarNet traces | WetLinks seconds release |
|---|---|---|
| 1 Hz throughput | yes | **yes**, verified |
| throughput is capacity, not offered load | yes | **yes**, iperf under load |
| sample count | 2.48M (USA) | **1.02M** (Osnabruck), 0.73M (Enschede) |
| 15 s scheduling phase recoverable | yes | **yes**, not aliased at 1 Hz |
| candidate satellite count | measured | **reconstructed**, exact from SGP4 |
| serving satellite elevation, distance | measured | **not recoverable** |
| weather | yes | yes, co-located station |

Verified rather than assumed: 68,596 iperf runs of exactly 15 samples each,
median 214.4 Mbps, spanning 179.7 days. BG-CFQS used 1,123,832 samples at CHI;
Osnabruck supplies 1,019,109. The median is a close match for StarNet's
reported "throughput holds around 230 Mbps", which is independent evidence that
these are genuine capacity measurements.

Satellite geometry is reconstructed by propagating 181 consecutive days of
Space-Track Starlink elements (2023-09-14 to 2024-03-12, no gaps) against the
known site coordinates. `ingest/wetlinks_full.attach_geometry` marks every row
`geometry_source = "reconstructed"` so the distinction cannot be lost
downstream.

## What the workaround costs, stated plainly

**Two of the four regime axes from CLAUDE.md section 7 are gone, and one of
them died for a reason worth recording.** Candidate count is exact: mean 37.6,
range 4 to 59, against StarNet's reported 15 to 45, and mean throughput rises
across its terciles (218.5, 221.2, 225.9 Mbps), which is the same direction
StarNet found, though far weaker than their 26%.

Serving satellite elevation and distance are *not* recoverable. Without
obstruction maps there is no way to tell which of the ~38 visible satellites
served the terminal. The obvious proxy, the highest satellite in view, is
useless: its elevation has standard deviation 3.9 degrees around 81, because
with that many satellites the best one is near zenith almost always, and its
correlation with throughput is -0.02. This is a negative result, not a missing
feature, and it is reported as one.

**The reproduction gates cannot be run.** StarNet's RMSE/MAE table and
BG-CFQS's average table both need the traces. Phase 1 and Phase 2 stay open,
and no number in this project should be compared to a published one. The 15
sample iperf run also caps look-back plus horizon at 15, so StarNet's 30/5 and
BG-CFQS's 75/15 are impossible here regardless.

**What survives is the part that matters.** The conditional risk failure that
motivates the whole project reproduces on this data without the traces: the
uncalibrated forecaster runs at OverRate 0.50 globally and 0.84 on the lowest
decile. BG-CFQS report 0.349 against 0.83 to 0.86. Same shape, independent
dataset, different continent, measured rather than quoted. The motivation
figure did not need their data after all.

## Still worth doing if the traces ever arrive

The elevation and distance axes, and therefore the honest version of the
objective O2 question. Everything else in the pipeline already runs and would
take the traces unchanged, because `ReplaySource` was built to their schema in
Phase 1.


---

# Unblocked, 2026-09-01

**The traces arrived.** Mayan obtained access to the OneDrive share. Everything
in the "Resolution" section above describes the period when they were
unavailable and is kept as the record of what was tried and what the fallback
cost; it is no longer the current state.

## What arrived, and what it actually was

Three files, one per location, plus `dataset_ho.pkl` and `sat_stats.npy`. The
filenames and the folders they were dropped into did not agree, so every file
was identified from its contents rather than its path:

| dropped as | rows | satellites | timezone | span | actually |
|---|---|---|---|---|---|
| `usa/Victoria Dataset.pkl` | 145,053 | 3,166 | -07:00 | Jul 11-28 | **Canada (VIC)** |
| `canada/dataset_tp_sat.pkl` | 1,123,832 | 5,723 | -05:00 | Apr 26-May 28 | **USA (CHI)** |
| `germany/Dataset TP Sat.pkl` | 613,295 | 3,956 | +02:00 | Jul 13-31 | **Germany (OSN)** |

USA and Canada were swapped. The identification is certain rather than inferred:
145,053 / 3,166 and 613,295 / 3,956 are exact matches to the paper's dataset
table, the timezones place the terminals on the right continents, and the date
ranges match BG-CFQS's three published processing windows.

Files now live at `data/starnet/{usa,canada,germany}/dataset_tp_sat.pkl`.
`dataset_ho.pkl` and `sat_stats.npy` were moved to `data/starnet/aux/`, because
`ReplaySource` concatenates every `*.pkl` in a location directory and
`dataset_ho` has a different schema.

## The US file is a subset, and that is correct

The paper reports collecting 2,475,163 US samples; the released file holds
1,123,832. Not a partial download:

- it is exactly the CHI sample count BG-CFQS report processing;
- `(1,123,832 - 45) / 46 + 1 = 24,430`, exactly the US data-point count StarNet
  report training on at sequence length 45 and step 46.

The released file is their training set. `replay.py` checks against
`RELEASED_FILE_STATS` for this reason and all three locations now verify at
zero relative error.

## Which file to ask for, if this ever has to be done again

`dataset_tp_sat.pkl`, one per location. It is the default path in
`main_pred.py`, `evaluate.py` and the TimesNet baseline's
`main_parameter_searching.py`. `dataset_nn.pkl` (524 MB) and
`aggregated_dataset_tp_sat.pkl` are never referenced by the model code, and the
8:2 split is done in code from the single file, so `dataset_tp_sat_test.pkl` is
not needed either.
