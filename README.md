# FlowConX

**Risk-controlled bandwidth allocation for Starlink access links, from the
research package to a running console.**

[![CI](https://github.com/Mayan10/flwcnx/actions/workflows/ci.yml/badge.svg)](https://github.com/Mayan10/flwcnx/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-248%20passing-brightgreen.svg)](newml/flwcnx/tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](newml/flwcnx/LICENSE)
[![Reproduction gates](https://img.shields.io/badge/reproduction%20gates-2%2F2%20passed-brightgreen.svg)](newml/flwcnx/docs/paper/report.md#4-reproduction-gates)

Mayan Sharma, Kriti Saini, Devansh Behl

---

A LEO access link's capacity moves on a 15 second scheduling cadence and varies
by more than a factor of three within an hour. An allocator acting on a point
forecast over-promises whenever the forecast is optimistic, and over-promising
drops sessions.

FlowConX forecasts throughput one horizon ahead, converts that forecast into a
**safe lower bound** whose overestimation rate is held at a risk budget the
operator sets, and drives admission control and congestion alerts from that
bound. The research that establishes what the bound can and cannot deliver is
in [`newml/flwcnx`](newml/flwcnx); the rest of this repository is the system
built around it.

## Repository layout

| Path | What it is | Stack |
|---|---|---|
| [`newml/flwcnx`](newml/flwcnx) | **The research package.** Forecasting, calibration, evaluation, the paper and every committed result. Start here. | Python 3.11+, PyTorch |
| [`newml/flwcnx/flwcnx/api`](newml/flwcnx/flwcnx/api) | Streams live decisions out of the real pipeline over WebSocket | FastAPI |
| [`backend`](backend) | Accounts, API keys and company records | Node, Express, Prisma, Postgres |
| [`webui`](webui) | Web console: landing page and the live decision dashboard | Next.js, TypeScript |
| [`tui`](tui) | Terminal console over the same decision stream | Rust, ratatui |
| [`docker-compose.yml`](docker-compose.yml) | The whole stack | Docker |

## How the pieces connect

```
                 newml/flwcnx  (the research package)
                 StarNet forecast -> regime -> safe bound
                 -> admission -> congestion
                              |
                       flwcnx.api.server
                              |
              ws://localhost:8010/ws/ml/stream
                       |              |
                   webui           tui          <- one decision per message
                       |
            http://localhost:3001/api           <- accounts and API keys
                       |
                    backend  ->  Postgres
```

Every decision on the stream comes out of the real stack rather than a mock.
The `--mode` flag only chooses which trace is replayed: `synthetic` generates a
trace carrying the 15 s period, the handover dip and the geometry effects and
needs no dataset, while `real` replays the recorded StarNet traces. Synthetic
output is for demonstration and is never reported as a result.

[`tui/connection.md`](tui/connection.md) documents the stream protocol in full.

## Quickstart

### The whole stack

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Web console | <http://localhost:3000> |
| Backend API | <http://localhost:3001/api> |
| ML API | <http://localhost:8010> |
| Postgres | `localhost:5433` |

The ML service trains its forecaster at startup, roughly a minute on CPU for
the synthetic trace. Until it is ready a connecting client receives a `status`
message and waits, rather than an error.

### The research package on its own

No Docker, no dataset and no GPU needed:

```bash
cd newml/flwcnx
python -m pip install -e ".[dev]"
pytest -q                                   # 248 tests, synthetic fixtures
```

### The ML API on its own

```bash
cd newml/flwcnx
python -m pip install -e ".[api]"
python -m flwcnx.api.server --mode synthetic --port 8010
```

### The terminal console

```bash
cd tui
cargo run
```

`h` home, `k` login, `m` the live ML dashboard, `/` command bar, `q` quit.

### The web console and backend in development

```bash
cd backend && npm install && npm run dev     # port 3001
cd webui   && npm install && npm run dev     # port 3000
```

The console's defaults already point at `localhost:3001/api` and
`localhost:8010`, so no environment file is needed for local development.
Override with `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_ML_WS_URL`, which Next
inlines at build time.

## The research, in one paragraph

Three of the six problems in the project brief are addressed, chosen because
they chain into one system rather than three disconnected models: predict
throughput degradation, detect congestion before users are affected, and
optimize bandwidth allocation automatically. Both reproduction gates pass, and
the findings are empirical rather than algorithmic: every mechanism evaluated
is published and cited as such. The headline results are that the state of the
art cannot serve a risk budget below 0.15, that its guarantee is conditional on
an exchangeable split a deployed terminal does not have, and that the satellite
covariates the project was designed around do not carry the signal.

**[Read the full README of the research package](newml/flwcnx/README.md)**, which
gives equal room to three findings and three failures, or go directly to:

| | |
|---|---|
| **Paper** | [`newml/flwcnx/docs/paper/paper.md`](newml/flwcnx/docs/paper/paper.md) |
| **Report** | [`newml/flwcnx/docs/paper/report.md`](newml/flwcnx/docs/paper/report.md) |
| **Results** | [`newml/flwcnx/results/summary/`](newml/flwcnx/results/summary/) |
| **Limitations** | [`newml/flwcnx/docs/limitations.md`](newml/flwcnx/docs/limitations.md) |

## Status

The research package is complete: both reproduction gates pass, all six of the
brief's industry needs are measured, and 248 tests run in CI on Python 3.11,
3.12 and 3.13.

The platform around it is a working demonstration rather than a deployment.
Stated plainly:

- There is **no live terminal**. `ingest/live.py` raises, and every decision on
  the stream is a replay of a recorded or generated trace.
- The backend's JWT secret and database password are development defaults
  committed in `docker-compose.yml`. They must be replaced before this is
  exposed to anything.
- `depends_on` in compose orders startup but does not wait for readiness, so
  the console may briefly show a disconnected stream while the ML service
  trains.
- The web and terminal consoles read the stream; neither can change the running
  configuration.

The research package's own limitations are the longer and more important list,
and they are in
[`newml/flwcnx/docs/limitations.md`](newml/flwcnx/docs/limitations.md), written
to be read before the results rather than after.

## Licence

MIT, see [`LICENSE`](newml/flwcnx/LICENSE). The licence covers the code only.
The datasets are redistributed by their own authors under their own terms, and
several components are reimplementations of published methods, identified as
such in their module docstrings. If you use this work, see
[`CITATION.cff`](newml/flwcnx/CITATION.cff) and cite the papers being
evaluated, listed in the research package's README.
