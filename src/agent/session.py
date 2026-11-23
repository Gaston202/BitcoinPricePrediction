"""Background trading sessions the dashboard starts, polls and stops.

Two modes:
  replay - walk through the unseen TEST window candle by candle (historical
           Binance data, honest out-of-sample), `speed` candles per second.
  live   - poll real Binance prices every few seconds; the hourly model makes a
           decision every `decision_seconds` on the still-forming candle.

Both are paper trading against a simulated USD/BTC wallet.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any

import pandas as pd

from src.agent import market as market_mod
from src.agent import model_service
from src.agent.trader import AgentConfig, TradingAgent
from src.features.engineering import feature_matrix

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 5


class TradingSession:
    def __init__(self, mode: str = "replay", interval: str = "1D", speed: float = 10.0,
                 agent_cfg: AgentConfig | None = None, poll_seconds: float = 3.0,
                 decision_seconds: float = 60.0, market=market_mod) -> None:
        if mode not in ("replay", "live"):
            raise ValueError("mode must be 'replay' or 'live'")
        if mode == "live":
            interval = "1h"  # a daily model would decide only once a day
        if interval not in model_service.INTERVALS:
            raise ValueError(f"interval must be one of {list(model_service.INTERVALS)}")
        self.mode, self.interval = mode, interval
        self.speed = min(max(float(speed), 0.2), 500.0)
        self.poll_seconds, self.decision_seconds = poll_seconds, decision_seconds
        self.market = market
        self.agent = TradingAgent(agent_cfg)
        self.status = "starting"
        self.message = "Loading model..."
        self.error: str | None = None
        self.total_steps: int | None = None
        self.history: list[dict[str, Any]] = []  # pre-session price context (live)
        self.started_at = pd.Timestamp.now(tz="UTC").isoformat()
        self._step = pd.Timedelta(interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="trading-session")

    # ------------------------------------------------------------- control
    def start(self) -> "TradingSession":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self.status in ("starting", "running"):
            self.status, self.message = "stopped", "Stopped by user"

    @property
    def running(self) -> bool:
        return self._thread.is_alive() and not self._stop.is_set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    # ------------------------------------------------------------- loops
    def _run(self) -> None:
        try:
            bundle = model_service.get_bundle(self.interval)
            if self._stop.is_set():
                return
            if self.mode == "replay":
                self._run_replay(bundle)
            else:
                self._run_live(bundle)
        except Exception as exc:  # surface any failure to the dashboard
            logger.exception("Trading session failed")
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
            self.message = self.error
            logger.debug(traceback.format_exc())

    def _record(self, timestamp, price, pred, allow_trade=True) -> None:
        with self._lock:
            rec = self.agent.step(timestamp, price, pred, allow_trade=allow_trade)
            rec["is_decision"] = allow_trade
            rec["forecast_for"] = (pd.Timestamp(timestamp) + self._step).isoformat()

    def _run_replay(self, bundle) -> None:
        clean, feats = model_service.test_window(self.interval)
        preds = bundle.predict(feats)
        self.total_steps = len(preds)
        self.status = "running"
        self.message = (f"Replaying unseen test window {preds.index[0]:%Y-%m-%d} -> "
                        f"{preds.index[-1]:%Y-%m-%d} ({len(preds)} {self.interval} candles)")
        delay = 1.0 / self.speed
        for open_time, pred in preds.iterrows():
            if self._stop.is_set():
                return
            # The decision is made when the candle closes, at its close price.
            self._record(open_time + self._step, clean.at[open_time, "close"], pred.to_dict())
            time.sleep(delay)
        self.status, self.message = "finished", "Replay finished - full test window traded"

    def _run_live(self, bundle) -> None:
        errors = 0
        last_decision = 0.0
        pred: dict | None = None
        seeded = False
        while not self._stop.is_set():
            try:
                now = time.time()
                if pred is None or now - last_decision >= self.decision_seconds:
                    klines = self.market.fetch_klines(self.interval, limit=120)
                    if not seeded:
                        self.history = [{"timestamp": (t + self._step).isoformat(), "price": float(c)}
                                        for t, c in klines["close"].iloc[:-1].items()]
                        seeded = True
                    feats = feature_matrix(self.market.provisional_frame(klines)).dropna()
                    if feats.empty:
                        raise RuntimeError("not enough candles to compute features")
                    pred = bundle.predict(feats.iloc[[-1]]).iloc[0].to_dict()
                    price = float(klines["close"].iloc[-1])
                    self._record(pd.Timestamp.now(tz="UTC"), price, pred, allow_trade=True)
                    last_decision = now
                else:
                    price = self.market.fetch_price()
                    self._record(pd.Timestamp.now(tz="UTC"), price, pred, allow_trade=False)
                errors = 0
                self.status = "running"
                self.message = (f"Live paper trading on Binance BTCUSDT - price every "
                                f"{self.poll_seconds:.0f}s, AI decision every {self.decision_seconds:.0f}s")
            except Exception as exc:
                errors += 1
                self.message = f"Market data error ({errors}/{MAX_CONSECUTIVE_ERRORS}): {exc}"
                logger.warning(self.message)
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    self.status, self.error = "error", self.message
                    return
            self._stop.wait(self.poll_seconds)

    # ------------------------------------------------------------- state
    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            records = self.agent.decisions[since:]
            summary = self.agent.summary()
            n = len(self.agent.decisions)
        return {
            "mode": self.mode, "interval": self.interval, "speed": self.speed,
            "status": self.status, "message": self.message, "error": self.error,
            "started_at": self.started_at, "total_steps": self.total_steps,
            "next_index": n, "records": records, "summary": summary,
            "history": self.history if since == 0 else [],
        }
