# Live demo runbook

Streams the forecaster, the risk calibration and the decision layer through a
terminal dashboard, one decision per second, on a synthetic trace that carries
the structure the model is supposed to find.

Every decision on the stream comes out of the real pipeline: StarNet forecast,
regime assignment, adaptive safe bound, admission control, congestion. The
`--mode` flag chooses only which trace is replayed. **Synthetic data is for
demonstration and is never reported as a result.**

## Before the room fills

Two terminals, and start terminal 1 at least two minutes early. The forecaster
trains at startup, which takes 20 to 60 seconds, and the dashboard's rolling
figures need about 30 decisions before they mean anything.

Use a wide terminal. The KPI row splits into six tiles and wants 120 columns or
more to stay readable.

### Terminal 1: the ML service

```bash
python -m pip install -e ".[api]"          # once
python -m thalweg.api.server --mode synthetic --port 8010
```

Wait for the log line saying the pipeline is ready, or check it:

```bash
curl -s localhost:8010/api/ml/status
```

`"model_loaded": true` means you are ready to present.

### Terminal 2: the dashboard

```bash
cd tui
cargo build --release                      # once, takes a few minutes
./target/release/thalweg
```

Press **`m`** for the ML dashboard. Keys: `h` home, `k` login, `m` dashboard,
`/` command bar, `q` quit.

## What to point at, in order

**The three lines.** Grey is the throughput that actually arrived, blue is the
point forecast, the bold line is the calibrated safe bound. The bound tracks
the forecast but sits deliberately below it. That gap is the risk layer doing
its job, and it is not a constant: watch it widen when the link gets harder to
predict.

**The red strip under the chart.** One mark per slot where the bound promised
more than the link delivered. Those are the decisions the risk budget is
counted over. They should be present but sparse.

**RISK RATE against UNCALIBRATED.** The two tiles are measured on identical
decisions. The first is the calibrated bound, the second is what an
uncalibrated allocator committing to the raw forecast would have achieved.
Over a full replay these land near 0.36 and 0.46. The tile is green while the
rate sits inside the budget plus a small tolerance.

**DROPPED.** Sessions admitted that the link could not carry. Over a full
replay the calibrated policy averages 0.26 against 0.37 uncalibrated, a 29%
reduction, at 98.8% utilisation.

**CONGESTION.** Fires when the bound sits below the committed rate for a
sustained window. Not a separate model: it is a decision rule on the same
bound, which is why it costs almost nothing. Roughly 9 episodes per replay at
the default 200 Mbps commitment.

## Saying the honest thing about the numbers

Expect a panel to ask, so say it first:

- **The rate moves around early on.** A few hundred decisions is a small
  sample. At a budget of 0.35 the standard error over 300 decisions is about
  0.028, so a reading of 0.30 or 0.41 is ordinary. The claim that the layer
  tracks its budget to within a point is about the full test split of 13,530
  decisions, not about one minute of a live stream.
- **This is a replay, not a live terminal.** `ingest/live.py` raises. There is
  no dish.
- **Synthetic mode means the data is generated, not the pipeline.** The trace
  carries the 15 s scheduling period, the handover dip, elevation, distance,
  candidate count and weather; everything downstream of it is the real model.
  Nothing here is a reported result.

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Dashboard says CONNECTING | Service still training | Wait, check `curl -s localhost:8010/api/ml/status` |
| CONNECTION shows ERR | Nothing on port 8010 | Confirm terminal 1 is running and on the right port |
| RISK RATE shows a dash | Fewer than 30 decisions buffered | Wait half a minute |
| Tiles look cramped | Terminal too narrow | Widen to 120 columns or more |
| Want a different link | | `--location usa|canada|germany` |
| Want a different budget | | `--epsilon 0.2`, the dashboard reads it from the handshake |
| Want congestion more or less often | | `--commitment 220` or `--commitment 180` |

## Running against the recorded traces instead

If `data/starnet/` is populated, the same demo runs on real measurements:

```bash
python -m thalweg.api.server --mode real --location canada --port 8010
```

Training takes longer. Everything else is identical.
