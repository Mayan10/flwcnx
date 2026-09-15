# FlowConX ML ↔ TUI Connection Architecture

## Overview

The FlowConX system connects three components for real-time satellite throughput monitoring:

```
┌──────────────────────┐      WebSocket        ┌──────────────────┐
│   ML API Server      │ ────────────────────  │   Rust TUI       │
│   (Python/FastAPI)   │  ws://localhost:8010  │   (ratatui)      │
│   Port 8010 (host)   │  /ws/ml/stream        │                  │
└──────────────────────┘                       └──────────────────┘
         │                                              │
    DemoEngine                                     HTTP REST
    (flwcnx pipeline)                          ┌──────────────────┐
                                               │  Node.js Backend │
                                               │  (Express)       │
                                               │  Port 3001       │
                                               └──────────────────┘
                                                  Auth / Company
```

## ML API Server (`newml/flwcnx/flwcnx/api/server.py`)

Every decision comes from the real newml pipeline: StarNet forecast → regime
assignment → adaptive safe bound → admission → congestion (`flwcnx.demo.DemoEngine`,
wired up in `newml/flwcnx/flwcnx/api/pipeline.py`). The mode only picks which trace
is replayed:

- **synthetic** (default): the generated trace from `flwcnx/ingest/synthetic.py`
  (15 s scheduling period, handover dip, satellite geometry). No data needed.
- **real**: the recorded StarNet traces under `data/starnet/<location>`.

The forecaster is trained once per (mode, location, epsilon) and cached. The server's
own configuration trains at startup (about 1 minute on CPU for synthetic); until it is
ready, a connecting client gets a `status` message and then waits for `init`. The test
split is replayed in a loop through one engine, so the stream never ends.

### Startup

```bash
# Docker (docker-compose.yml builds ./newml, host port 8010 -> container 8000)
docker compose up -d ml-api

# Local
cd newml/flwcnx && pip install -e ".[api]"
python -m flwcnx.api.server --mode synthetic --port 8010
python -m flwcnx.api.server --mode real --location canada --port 8010
```

Options: `--epsilon` (0.35), `--epochs` (5), `--stride` (2), `--commitment` (150 Mbps),
`--location` (canada). `--synthetic` is kept as shorthand for `--mode synthetic`.

### REST Endpoints

| Method | Path              | Description                     |
|--------|-------------------|---------------------------------|
| GET    | `/health`         | Liveness probe                  |
| GET    | `/api/ml/status`  | Engine state, training/ready pipelines, counters |
| GET    | `/api/ml/snapshot` | Latest DecisionRecord as JSON  |

### WebSocket Endpoint: `/ws/ml/stream`

**Handshake**: On connect, the client may send an optional JSON config message within 2 seconds:

```json
{
  "location": "canada",
  "epsilon": 0.35,
  "speed": 1.0,
  "mode": "synthetic",
  "stride": 2,
  "steps": 0
}
```

If no config is received, defaults are used.

**Server → Client messages**:

0. **Status** (only if the requested pipeline is still training):
```json
{ "type": "status", "state": "training", "mode": "synthetic", "location": "canada" }
```

1. **Init** (sent once, when the pipeline is ready):
```json
{
  "type": "init",
  "mode": "synthetic",
  "location": "canada",
  "epsilon": 0.35,
  "direction": "lower",
  "feature_set": "throughput_mbps,sat_id_encoded,elevation_deg,...",
  "n_test_samples": 598
}
```

2. **Decision** (sent at ~1 Hz):
```json
{
  "type": "decision",
  "step": 42,
  "timestamp": "2024-01-15T12:34:56+00:00",
  "regime": "level=L1",
  "predicted_mbps": 185.234,
  "bound_mbps": 142.891,
  "actual_mbps": 178.456,
  "offset_mbps": -42.343,
  "alpha": 0.3412,
  "admitted": 14,
  "oracle": 17,
  "dropped": 0,
  "unused": 3,
  "risk_event": false,
  "congested": false,
  "realised_risk_rate": 0.0238
}
```

3. **Done** (only when the client asked for a finite `steps`):
```json
{ "type": "done" }
```

4. **Error**:
```json
{
  "type": "error",
  "message": "description of the problem"
}
```

## DecisionRecord Fields

| Field                 | Type    | Description                                                    |
|-----------------------|---------|----------------------------------------------------------------|
| `step`                | int     | Zero-indexed decision slot number                              |
| `timestamp`           | string  | ISO 8601 timestamp of the forecast origin                      |
| `regime`              | string  | Operating regime label (e.g., `level=L0` … `level=L3`)          |
| `predicted_mbps`      | float   | Raw ML point forecast (Mbps)                                   |
| `bound_mbps`          | float   | Risk-calibrated safe lower bound (Mbps) — **primary metric**   |
| `actual_mbps`         | float   | Ground truth observed throughput (Mbps)                        |
| `offset_mbps`         | float   | Additive offset applied: `bound = predicted + offset`          |
| `alpha`               | float   | Current operating point of the adaptive calibrator             |
| `admitted`            | int     | Sessions admitted: `floor(bound / 10)`                         |
| `oracle`              | int     | Sessions possible with perfect foresight: `floor(actual / 10)` |
| `dropped`             | int     | Over-admitted sessions: `max(admitted - oracle, 0)`            |
| `unused`              | int     | Under-utilized sessions: `max(oracle - admitted, 0)`           |
| `risk_event`          | bool    | `true` when `bound > actual` (over-promise)                    |
| `congested`           | bool    | `true` when bound < commitment for 5+ consecutive slots        |
| `realised_risk_rate`  | float   | Running average of risk events so far                          |

## TUI Client (`tui/src/ml_client.rs`)

The TUI spawns a background tokio task on startup that:

1. Connects to `ws://127.0.0.1:8010/ws/ml/stream` (override with `FLWCNX_API_URL` env var).
2. Deserializes each JSON message into Rust structs.
3. Sends `MlMessage` values through an mpsc channel to the app event loop.
4. On disconnect, retries with exponential backoff (1s → 2s → 4s → ... → 30s max).

## TUI State Management

In `app.rs`, the ML data is stored in fixed-size circular buffers (`VecDeque<f64>`, capacity 300 = ~5 minutes at 1 Hz):

- `ml_actual` — actual throughput history
- `ml_predicted` — point forecast history
- `ml_bound` — safe bound history
- `ml_alpha` — alpha (operating point) history
- `ml_admitted` / `ml_dropped` / `ml_unused` — admission metrics history

The event loop drains `ml_rx` every tick (150ms) and pushes new values into the buffers.

## TUI Navigation

- Press `m` or type `/ml` to navigate to the ML Monitor dashboard.
- Press `l` or type `/live` for the existing traffic classifier demo dashboard.
- The ML Monitor does **not** require authentication (it reads from the local ML API, not the Node.js backend).

## Startup Sequence

1. Start the ML API server:
   ```bash
   docker compose up -d ml-api
   # or: cd newml/flwcnx && python -m flwcnx.api.server --mode synthetic --port 8010
   ```
2. Start the backend (for auth, if needed):
   ```bash
   cd backend && npm run dev
   ```
3. Start the TUI:
   ```bash
   cd tui && cargo run
   ```
4. Press `m` to open the ML Monitor dashboard.

## Environment Variables

| Variable          | Default                             | Description                      |
|-------------------|-------------------------------------|----------------------------------|
| `FLWCNX_API_URL`  | `ws://127.0.0.1:8010/ws/ml/stream` | ML WebSocket server URL (TUI)    |
| `FLWCNX_DEVICE`   | `auto`                              | PyTorch device (cpu/cuda/mps)    |
