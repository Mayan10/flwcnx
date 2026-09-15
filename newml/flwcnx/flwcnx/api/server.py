#!/usr/bin/env python3
"""FastAPI server exposing the flwcnx pipeline over WebSocket and REST.

Every decision on the stream comes out of the real stack: StarNet forecast ->
regime assignment -> adaptive safe bound -> admission -> congestion, via
`demo.DemoEngine`. The mode only picks the trace it replays (`api/pipeline.py`):

  synthetic   the generated trace from `ingest/synthetic.py`. No data needed.
  real        the recorded StarNet traces under `data/starnet/<location>`.

The forecaster is trained once per (mode, location, epsilon) and cached, so the
first client for a configuration waits for training and every later one starts
immediately. The server's own configuration is trained at startup.

Endpoints
---------
GET  /health                 Liveness probe.
GET  /api/ml/status          Engine state, including whether training is done.
GET  /api/ml/snapshot        Latest DecisionRecord as JSON.
WS   /ws/ml/stream           Real-time decision stream (primary).

Usage
-----
    python -m flwcnx.api.server --mode synthetic --port 8000
    python -m flwcnx.api.server --mode real --location canada --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import time
from dataclasses import replace

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from flwcnx.api.pipeline import PipelineSpec, TrainedPipeline, build_pipeline

logger = logging.getLogger("flwcnx.api")

MODES = ("synthetic", "real")
LOCATIONS = ("usa", "canada", "germany")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class EngineState:
    """Shared state: the default spec, trained pipelines, and counters."""

    def __init__(self) -> None:
        self.spec = PipelineSpec()
        self.pipelines: dict[PipelineSpec, TrainedPipeline] = {}
        self.building: dict[PipelineSpec, asyncio.Task] = {}
        self.last_error: str | None = None
        self.total_decisions: int = 0
        self.active_streams: int = 0
        self.latest_record: dict | None = None
        self.started_at: float = time.time()

    async def pipeline(self, spec: PipelineSpec) -> TrainedPipeline:
        """Return the trained pipeline for `spec`, training it at most once
        however many clients ask for it concurrently."""
        if spec in self.pipelines:
            return self.pipelines[spec]
        task = self.building.get(spec)
        if task is None:
            task = asyncio.create_task(asyncio.to_thread(build_pipeline, spec))
            self.building[spec] = task
        try:
            pipeline = await asyncio.shield(task)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if task.done():
                self.building.pop(spec, None)
        self.pipelines[spec] = pipeline
        self.last_error = None
        return pipeline

    def status(self) -> dict:
        return {
            "mode": self.spec.mode,
            "location": self.spec.location,
            "epsilon": self.spec.epsilon,
            "model_loaded": self.spec in self.pipelines,
            "training": [f"{s.mode}/{s.location}/{s.epsilon}" for s in self.building],
            "ready": [f"{s.mode}/{s.location}/{s.epsilon}" for s in self.pipelines],
            "last_error": self.last_error,
            "active_streams": self.active_streams,
            "total_decisions": self.total_decisions,
            "uptime_seconds": round(time.time() - self.started_at, 1),
        }


_state = EngineState()


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    # Train the default configuration in the background so the first client
    # does not pay for it, without holding up the HTTP server's startup.
    warmup = asyncio.create_task(_state.pipeline(_state.spec))
    warmup.add_done_callback(
        lambda t: t.cancelled() or t.exception() is None
        or logger.error("Startup training failed: %s", t.exception()))
    yield


app = FastAPI(
    title="FlowConX ML API",
    description="Real-time throughput forecasting and risk-controlled allocation.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _state.spec in _state.pipelines,
            "mode": _state.spec.mode}


@app.get("/api/ml/status")
async def ml_status():
    return JSONResponse(_state.status())


@app.get("/api/ml/snapshot")
async def ml_snapshot():
    if _state.latest_record is None:
        return JSONResponse({"error": "no decisions yet"}, status_code=404)
    return JSONResponse(_state.latest_record)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

def _spec_from_client(config: dict) -> PipelineSpec:
    """Overlay the client's optional handshake onto the server's defaults."""
    spec = _state.spec
    mode = config.get("mode", spec.mode)
    location = config.get("location", spec.location)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if location not in LOCATIONS:
        raise ValueError(f"location must be one of {LOCATIONS}, got {location!r}")
    return replace(spec, mode=mode, location=location,
                   epsilon=float(config.get("epsilon", spec.epsilon)),
                   stride=int(config.get("stride", spec.stride)))


@app.websocket("/ws/ml/stream")
async def ws_ml_stream(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket client connected")
    _state.active_streams += 1

    try:
        # Optional config as the first message; a client that sends nothing
        # gets the server's defaults after two seconds.
        config: dict = {}
        with contextlib.suppress(asyncio.TimeoutError, json.JSONDecodeError):
            config = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=2.0))

        spec = _spec_from_client(config)
        speed = max(float(config.get("speed", 1.0)), 0.1)
        steps = int(config.get("steps", 0))  # 0 = forever

        if spec not in _state.pipelines:
            await ws.send_json({"type": "status", "state": "training",
                                "mode": spec.mode, "location": spec.location})
        pipeline = await _state.pipeline(spec)
        await ws.send_json(pipeline.init_message())

        interval = 1.0 / speed
        for record in pipeline.stream(steps):
            data = record.to_dict() | {"type": "decision"}
            _state.latest_record = data
            _state.total_decisions += 1
            await ws.send_json(data)
            await asyncio.sleep(interval)

        await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": str(exc)})
    finally:
        _state.active_streams -= 1


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="FlowConX ML API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", default="synthetic", choices=MODES)
    parser.add_argument("--synthetic", action="store_true",
                        help="Shorthand for --mode synthetic")
    parser.add_argument("--location", default="canada", choices=LOCATIONS)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=PipelineSpec.epochs)
    parser.add_argument("--stride", type=int, default=PipelineSpec.stride)
    parser.add_argument("--commitment", type=float, default=PipelineSpec.commitment_mbps)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    _state.spec = PipelineSpec(
        mode="synthetic" if args.synthetic else args.mode,
        location=args.location, epsilon=args.epsilon, epochs=args.epochs,
        stride=args.stride, commitment_mbps=args.commitment,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
