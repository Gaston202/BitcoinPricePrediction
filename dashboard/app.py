"""Admin dashboard: project charts, live data sync and the rule-based multi-agent trader.

Run from the project root:
    python -m dashboard            # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.agents.config import DEFAULT_CONFIG, INTERVAL_PRESETS
from src.agents.overview import build_overview
from src.agents.session import LIVE_INTERVALS, PERIODS, REPLAY_INTERVALS, TradingSession
from src.data import binance, update

logger = logging.getLogger("dashboard")
STATIC_DIR = Path(__file__).parent / "static"
SYNC_EVERY = float(os.getenv("BTC_SYNC_EVERY", "300"))
SYNC_ENABLED = os.getenv("BTC_SYNC", "1") != "0"


def _clean(obj: Any) -> Any:
    """JSON can't carry NaN/inf; send null instead."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return _clean(obj.item())
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def ok(payload: Any) -> JSONResponse:
    return JSONResponse(_clean(payload))


# ---------------------------------------------------------------- overview (rebuilt when data changes)
_overview: dict[str, Any] = {"status": "loading"}
_overview_lock = threading.Lock()


def _load_overview() -> None:
    global _overview
    if not _overview_lock.acquire(blocking=False):
        return  # a rebuild is already running
    try:
        data = build_overview()
        _overview = {"status": "ready", **data}
    except Exception as exc:
        logger.exception("Overview failed")
        _overview = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        _overview_lock.release()


updater = update.BackgroundUpdater(every=SYNC_EVERY)
updater.on_new_data(lambda _status: threading.Thread(target=_load_overview, daemon=True).start())
_session: TradingSession | None = None
_session_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_load_overview, daemon=True).start()
    if SYNC_ENABLED:
        updater.start()
    yield
    updater.stop()
    if _session is not None:
        _session.stop()


app = FastAPI(title="BTC Multi-Agent Trader", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if config.FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=config.FIGURES_DIR), name="figures")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/overview")
def overview() -> JSONResponse:
    return ok(_overview)


@app.post("/api/overview/refresh")
def refresh_overview() -> JSONResponse:
    threading.Thread(target=_load_overview, daemon=True).start()
    return ok({"status": "loading"})


# ---------------------------------------------------------------- data sync
@app.get("/api/data/status")
def data_status() -> JSONResponse:
    st = updater.status if SYNC_ENABLED else update.read_status()
    return ok({**st.__dict__, "syncing": updater.running, "auto_sync": SYNC_ENABLED,
               "every_seconds": SYNC_EVERY})


@app.post("/api/data/sync")
def data_sync() -> JSONResponse:
    if SYNC_ENABLED:
        updater.sync_now()
        return ok({"started": True})
    status = update.sync()
    if status.added:
        threading.Thread(target=_load_overview, daemon=True).start()
    return ok({"started": True, **status.__dict__})


# ---------------------------------------------------------------- agent sessions
class StartRequest(BaseModel):
    mode: Literal["replay", "live"] = "replay"
    interval: Literal["1m", "5m", "15m", "1h", "4h", "1D"] = "4h"
    period: Literal["new", "30d", "90d", "180d", "365d", "test"] = "90d"
    speed: float = Field(20.0, gt=0, le=500)
    initial_cash: float = Field(1000.0, gt=0, le=1e9)
    risk_profile: Literal["aggressive", "neutral", "conservative"] = "neutral"
    analysts: list[Literal["trend", "momentum", "volatility", "volume", "sentiment", "quant"]] = Field(
        default_factory=lambda: list(DEFAULT_CONFIG["analysts"]), min_length=1)
    max_debate_rounds: int = Field(2, ge=1, le=4)
    max_risk_rounds: int = Field(1, ge=1, le=3)


@app.get("/api/agents/config")
def agents_config() -> JSONResponse:
    return ok({"defaults": DEFAULT_CONFIG, "interval_presets": INTERVAL_PRESETS,
               "replay_intervals": REPLAY_INTERVALS, "live_intervals": LIVE_INTERVALS, "periods": list(PERIODS)})


@app.post("/api/session/start")
def start_session(req: StartRequest) -> JSONResponse:
    global _session
    with _session_lock:
        if _session is not None:
            _session.stop()
        cfg = {"initial_cash": req.initial_cash, "risk_profile": req.risk_profile, "analysts": list(req.analysts),
               "max_debate_rounds": req.max_debate_rounds, "max_risk_rounds": req.max_risk_rounds}
        try:
            _session = TradingSession(mode=req.mode, interval=req.interval, period=req.period,
                                      speed=req.speed, cfg=cfg).start()
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return ok(_session.snapshot())


@app.post("/api/session/stop")
def stop_session() -> JSONResponse:
    if _session is None:
        raise HTTPException(404, "no session")
    _session.stop()
    return ok(_session.snapshot(since=10**9))


@app.get("/api/session/state")
def session_state(since: int = 0) -> JSONResponse:
    if _session is None:
        return ok({"status": "idle"})
    return ok(_session.snapshot(since=max(since, 0)))


@app.get("/api/session/step/{i}")
def session_step(i: int) -> JSONResponse:
    if _session is None or (state := _session.state_at(i)) is None:
        raise HTTPException(404, "no stored state for this step")
    return ok(state)


@app.get("/api/market/price")
def market_price() -> JSONResponse:
    try:
        return ok({"price": binance.fetch_price()})
    except binance.MarketDataError as exc:
        raise HTTPException(503, str(exc)) from exc
