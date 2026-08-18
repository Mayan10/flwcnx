# The full WetLinks release

https://github.com/sys-uos/WetLinks (Laniewski et al., TMA 2024, entry 9).

The supplied CSVs were a subset carrying only the 30 s dish status stream. The
public repository holds considerably more, and it changes what this project can
do. Fetched with `scripts/download_data.py --dataset wetlinks`.

## What is in it

| File | Size | What it is |
|---|---|---|
| `iperf_cleaned_seconds_*.csv` | 70 / 99 MB | **per-second iperf throughput** |
| `analysis_data_*.csv` | 14 / 22 MB | per-run means plus weather-station readings |
| `iperf_cleaned_means_*.csv` | 7 / 10 MB | per-run throughput means |
| `Raw_Data/` | | raw per-site captures |
| `Download_Links.txt` | | DWD and KNMI weather station sources |

Sites are Osnabrück (`uos-rz`) and Enschede (`utwente`), September 2023 to
March 2024.

## Why the per-second file matters

```
site_name, timestamp_start,     timestamp_end,       download
uos-rz,    2023-09-14 15:12:22, 2023-09-14 15:12:23, 142011000.0
uos-rz,    2023-09-14 15:12:23, 2023-09-14 15:12:24, 162277000.0
```

1 Hz, and the number is what an iperf transfer actually achieved, so it is
capacity rather than whatever traffic happened to be flowing. Both limitations
the supplied subset imposed are lifted:

- **the scheduling phase is recoverable.** 30 s sampling aliased it to a
  constant; 1 Hz does not. `state/phase.py` becomes usable, and the phase
  regime axis comes back.
- **the target is a real capacity figure**, so the lower-bound framing, the
  admission control layer and the comparison against StarNet and BG-CFQS all
  mean what they are supposed to mean.

Samples arrive in bursts: an iperf run is contiguous at 1 Hz, and runs are
about 3 minutes apart. Windows must not span a gap between runs, so the
segmenter uses a 3 second threshold.

## Weather, and a sentinel trap

`analysis_data_*.csv` carries `temp, humidity, dewpt, windchill, winddir,
windspeed, windgust, rain, solarradiation, uv, barom` from stations co-located
with the terminals. Better than the Open-Meteo reconstruction that was the
fallback, and it makes the WetLinks weather-prediction result (entry 10)
directly comparable.

The trap: those columns carry `-9999` style missing markers, and the upstream
preprocessing averaged them together with real readings, so the contaminated
values are not a single magic number. They appear as `-9999`, `-6666`, `-3333`,
`-5572.78`, `-4469.94` and others, depending on how many sentinels went into
each average.

At Osnabrück this affects **61 of 68,597 rows (0.09%)**. Small, but those rows
are *not missing at random*: their median throughput is 190.3 Mbps against
212.3 Mbps for the rest. So they are masked to NaN rather than dropped, and the
mask is by physical plausibility per variable rather than by matching known
sentinel values, which also catches the partially-averaged cases. See
`WEATHER_BOUNDS` in `ingest/wetlinks_full.py`.

Left unhandled, a mean windspeed of -1.61 m/s and a mean temperature of 3.85 C
against a true 8.49 C would have gone into the feature matrix.

## What is still missing

No serving-satellite identity, elevation, distance or candidate count, in any
file. That has to come from propagated orbital elements
(`ingest/spacetrack.py`), which makes it a reconstruction and not a
measurement, and anything derived from it must be labelled that way. The
"highest satellite in view" is a *proxy* for the serving one; WetLinks does not
record which satellite served.

## Network notes

Recorded because they cost time. From this machine:

- `celestrak.org` does not respond at all, so Space-Track is the only element
  source. It was needed anyway for historical elements.
- `raw.githubusercontent.com` resets the connection on large transfers, and the
  GitHub contents API fails above roughly 22 MB. Ranged GETs work, so the large
  files are fetched in 4 MB chunks with per-chunk retries and resume.
- `github.com`, `api.github.com`, `open-meteo.com` and `space-track.org` all
  respond normally.
