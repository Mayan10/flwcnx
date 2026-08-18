# Datasets

## Primary: StarNet traces

Repo: https://github.com/ConnectedSystemsLab/StarNet
Paper: https://dl.acm.org/doi/10.1145/3768971

### The data is not in the repo

The repo carries the measurement tool, the model and a TimesNet baseline. The
traces themselves are on OneDrive, one link per country, and there is no
public direct-download URL. They have to be fetched by hand:

- US: https://uillinoisedu-my.sharepoint.com/:f:/g/personal/zikunliu_illinois_edu/IgBaSUJKNIjlQI6KeF1tWc6-AZMwiOlyueqSY4KzOYsVEfw
- Germany: https://uillinoisedu-my.sharepoint.com/:f:/g/personal/zikunliu_illinois_edu/IgBfbwMBpJDTSqbqvty7kK52AalnjRwNyiW-Xomy-iUD01A
- Canada: https://uillinoisedu-my.sharepoint.com/:f:/g/personal/zikunliu_illinois_edu/IgDD3A4WdkPjQ4ME-CCIpk0GAfO40d2gSYm2k5skQMJVf-E

`scripts/download_data.py` clones the code and prints these links; it cannot
fetch the traces for you.

Put the downloaded folders at:

```
data/starnet/usa/dataset_tp_sat.pkl
data/starnet/canada/dataset_tp_sat.pkl
data/starnet/germany/dataset_tp_sat.pkl
```

### The real schema

The cleaned file is `dataset_tp_sat.pkl`, a pickled pandas DataFrame, not CSV.
Read off their `get_data_loader` in `model/NN_TP_ours/sat_dataset.py`, the
columns actually used are:

| upstream | ours | note |
|---|---|---|
| `timestamp` | `timestamp` | datetime64 |
| `throughput` | `throughput_mbps` | the target |
| `latency` | `latency_ms` | **present**, excluded from model inputs |
| `alt` | `elevation_deg` | altitude angle, not named "elevation" |
| `az` | `azimuth_deg` | |
| `distance` | `distance_km` | |
| `sat_name` | `sat_id` | string, label encoded downstream |
| `n_candidates` | `candidate_count` | |
| `clouds` | `cloud_cover_pct` | |
| `pressure` | `pressure_hpa` | |
| `humidity` | `humidity_pct` | **humidity, not precipitation** |
| `hour` | derived | |
| `location` | | only in the combined `data_all` file |

Two corrections to earlier assumptions this forced:

1. The traces carry **latency**. CLAUDE.md section 6 states they are throughput
   only, and on that basis calls the Casparsen crossover a stretch. It is
   cheaper than assumed: the latency column is already there. The 1 Hz
   sampling rate is still the real obstacle for the boundary analysis, not the
   absence of the signal.
2. The weather variable is **humidity**, not precipitation. BG-CFQS list
   humidity in their auxiliary variables and are consistent with the data;
   CLAUDE.md section 5 lists precipitation. `precipitation_mm` is retained in
   the normalized schema because Open-Meteo supplies it on the live path, but
   it is not a StarNet feature.

### The feature set, pinned

Their loader excludes `['timestamp', 'latency', 'throughput']` from the
attribute channels, leaving exactly twelve:

```
alt, az, distance, sat_name, n_candidates,
clouds, pressure, humidity,
t_s, t_minute, t_hour, t_d_of_w
```

with `throughput` carried as its own channel, so thirteen inputs in total. That
twelve is exactly the auxiliary list BG-CFQS report, which is the strongest
available evidence that both papers are working off the same columns.

CLAUDE.md section 6 says "input 11 features". The repo says twelve attributes
plus throughput. We follow the data, and `config.FEATURE_COLUMNS` holds the
thirteen. The discrepancy is recorded rather than resolved by picking whichever
number is convenient.

`t_s` is their 15 second phase, computed as `(timestamp.dt.second - 12) % 15`,
which hardcodes the 12/27/42/57 offset. We recover it instead
(`state/phase.py`) and fall back to 12 only when recovery is not confident.
Their `assign_chunk` helper hardcodes the same boundaries.

### Other divergences from CLAUDE.md worth knowing

- `main_pred.py` defaults to `hidden_size=60`, not the 128 in CLAUDE.md
  section 6. `n_fea_expanded=12`, `positional_scale=213`. Our
  `StarNetConfig` keeps 128 because that is what the paper text says; if the
  reproduction misses, the hidden size is the first thing to try.
- `main.py` runs `input_len=75, output_len=15`, which is the BG-CFQS look-back
  and horizon rather than the 30/5 headline configuration.
- Step lengths per location match CLAUDE.md: chi 46, vic 6, osn 29.
- **Their `SatelliteSequenceDataset_Continous` fits its `MinMaxScaler` on the
  whole trace before the train/validation split.** That leaks the validation
  distribution's scale into training. We fit the standardiser on training
  windows only (`state/features.py`), so our reproduction may land slightly
  worse than published for a reason that is not a bug on our side. Worth
  checking explicitly if the gate is missed by a small margin.

### Published statistics

Used by `ReplaySource.verify_against_published` as a loader self-check.

| | USA | Canada | Germany |
|---|---|---|---|
| Duration | 6 months | 1 month | 1 month |
| Trace minutes | 41,252 | 2,417 | 10,221 |
| Throughput samples | 2,475,163 | 145,053 | 613,295 |
| Unique serving satellites | 6,052 | 3,166 | 3,956 |
| Handovers | 86,808 | 7,257 | 26,782 |

BG-CFQS rename these CHI (US), OSN (Germany), VIC (Canada) and process
2024-04-26 to 2024-05-28 (CHI, 1,123,832 samples), 2024-07-13 to 2024-07-31
(OSN), 2024-07-11 to 2024-07-28 (VIC). Set `DataConfig.date_start/date_end` to
match when running the baseline comparison.

## Secondary: Horizon

Code: https://github.com/spear-lab/Horizon-Predicting-Starlink-Performance
DOI: https://doi.org/10.4121/0bf59468-e5cb-433f-aeb2-e04cf694b65c

Crowdsourced M-Lab NDT7 plus Cloudflare AIM, January to November 2025, 90+
countries, about 15.6M NDT7 speedtests and 157K AIM measurements, Starlink
identified by AS14593. Both accessible via BigQuery.

Hourly aggregated with no terminal telemetry, so it cannot support the
calibration work. Used for the O1 cross location analysis only.

## Supplied dataset

**Still an open question.** Nothing has appeared under `data/supplied/`. If it
does, run `python scripts/download_data.py --inspect data/supplied` and report
the schema before any feature code is written against it.
