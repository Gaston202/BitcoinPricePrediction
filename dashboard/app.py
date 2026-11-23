"""Admin dashboard: project charts + AI paper-trading control room.

Run from the project root:
    python -m dashboard            # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import math
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.agent import market, model_service
from src.agent.overview import build_overview
from src.agent.session import TradingSession
from src.agent.trader import AgentConfig

logger = logging.getLogger("dashboard")
STATIC_DIR = Path(__file__).parent / "static"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Build the overview in the background so the page loads instantly.
    threading.Thread(target=_load_overview, daemon=True).start()
    yield
    if _session is not None:
        _session.stop()


app = FastAPI(title="BTC AI Trading Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if config.FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=config.FIGURES_DIR), name="figures")


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
    return obj


def ok(payload: Any) -> JSONResponse:
    return JSONResponse(_clean(payload))


# ---------------------------------------------------------------- overview
_overview: dict[str, Any] = {"status": "loading"}


def _load_overview() -> None:
    global _overview
    try:
        _overview = {"status": "ready", **build_overview()}
    except Exception as exc:
        logger.exception("Overview failed")
        _overview = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/overview")
def overview() -> JSONResponse:
    return ok(_overview)


@app.post("/api/overview/refresh")
def refresh_overview() -> JSONResponse:
    global _overview
    _overview = {"status": "loading"}
    threading.Thread(target=_load_overview, daemon=True).start()
    return ok(_overview)


# ---------------------------------------------------------------- trading
class StartRequest(BaseModel):
    mode: Literal["replay", "live"] = "replay"
    interval: Literal["1D", "1h"] = "1D"
    speed: float = Field(10.0, gt=0, le=500)
    initial_cash: float = Field(1000.0, gt=0, le=1e9)
    stop_loss_pct: float | None = Field(5.0, gt=0, lt=100)
    take_profit_pct: float | None = Field(None, gt=0, lt=100)
    min_confidence_pct: float = Field(0.0, ge=0, le=100)
    fee_aware: bool = False
    decision_seconds: float = Field(60.0, ge=5, le=3600)


_session: TradingSession | None = None
_session_lock = threading.Lock()


@app.post("/api/session/start")
def start_session(req: StartRequest) -> JSONResponse:
    global _session
    with _session_lock:
        if _session is not None:
            _session.stop()
        cfg = AgentConfig(
            initial_cash=req.initial_cash,
            stop_loss=req.stop_loss_pct / 100 if req.stop_loss_pct else None,
            take_profit=req.take_profit_pct / 100 if req.take_profit_pct else None,
            min_confidence=req.min_confidence_pct / 100,
            fee_aware=req.fee_aware,
        )
        try:
            _session = TradingSession(mode=req.mode, interval=req.interval, speed=req.speed,
                                      agent_cfg=cfg, decision_seconds=req.decision_seconds).start()
        except ValueError as exc:
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


@app.get("/api/models")
def models_status() -> JSONResponse:
    out = {}
    for interval in model_service.INTERVALS:
        path = model_service.MODELS_DIR / f"agent_rf_{interval}.joblib"
        out[interval] = {"cached": path.exists()}
    return ok(out)


@app.get("/api/market/price")
def market_price() -> JSONResponse:
    try:
        return ok({"price": market.fetch_price()})
    except market.MarketDataError as exc:
        raise HTTPException(503, str(exc)) from exc
