"""Background replay / live sessions of the agent graph, polled by the dashboard.

replay  walks through stored candles (snapshot + every synced Binance update) one by one.
        Periods: the new data since the frozen snapshot, the last N days, or the team's
        unseen test window. Indicators are computed on the full history, so the first
        candle of the period already has its 200-candle context.
live    polls Binance: the price every few seconds (wallet marked to market, stop-loss
        enforced intrabar) and the full agent pipeline each time a candle closes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

import pandas as pd

from src import config as project_config
from src.agents import quant
from src.agents.config import config_for
from src.agents.graph import TradingAgentsGraph
from src.agents.market_data import WARMUP_BARS, attach_news, attach_sentiment, compute_indicators
from src.agents.news_sentiment import best_available
from src.data import binance as binance_mod
from src.data import update

logger = logging.getLogger(__name__)

REPLAY_INTERVALS = ("1h", "4h", "1D")
LIVE_INTERVALS = ("1m", "5m", "15m", "1h", "4h")
PERIODS = {"new": None, "30d": 30, "90d": 90, "180d": 180, "365d": 365, "test": "2025-07-09"}
BINANCE_INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h"}
MAX_CONSECUTIVE_ERRORS = 5
KEEP_STATES = 600


def period_start(period: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    if period not in PERIODS:
        raise ValueError(f"period must be one of {list(PERIODS)}")
    end = index.max()
    if period == "new":
        return project_config.CUTOFF_DATE + pd.Timedelta("1h")
    if period == "test":
        return pd.Timestamp(PERIODS["test"], tz="UTC")
    return end - pd.Timedelta(days=PERIODS[period])


def sentiment_series(live_ratio: bool = False, market=binance_mod) -> dict:
    out = {"funding": update.load_funding(), "fear_greed": update.load_fear_greed(), "long_short": None}
    if live_ratio:
        try:
            out["long_short"] = market.fetch_long_short_ratio()["long_short_ratio"]
        except Exception as exc:  # optional
            logger.info("long/short ratio unavailable: %s", exc)
    return out


def add_quant(ind: pd.DataFrame, cutoff: pd.Timestamp, hourly: pd.DataFrame | None = None) -> str:
    """Fill ind['p_up'] with the LightGBM probabilities (1h candles only). Returns a status note."""
    try:
        hourly = update.load_latest("1h") if hourly is None else hourly
        feats = quant.hourly_features(hourly)
        model = quant.get_model(hourly, cutoff, feats)
        ind["p_up"] = model.p_up(feats).reindex(ind.index)
        return f"quant model trained on {model.n_train:,} candles before {model.cutoff:%Y-%m-%d}"
    except Exception as exc:
        logger.warning("Quant analyst unavailable: %s", exc)
        ind["p_up"] = float("nan")
        return f"quant analyst off ({type(exc).__name__})"


class TradingSession:
    def __init__(self, mode: str = "replay", interval: str = "4h", period: str = "90d", speed: float = 20.0,
                 cfg: dict | None = None, poll_seconds: float = 3.0, market=binance_mod) -> None:
        if mode not in ("replay", "live"):
            raise ValueError("mode must be 'replay' or 'live'")
        allowed = REPLAY_INTERVALS if mode == "replay" else LIVE_INTERVALS
        if interval not in allowed:
            raise ValueError(f"{mode} interval must be one of {list(allowed)}")
        if mode == "replay" and period not in PERIODS:
            raise ValueError(f"period must be one of {list(PERIODS)}")
        self.mode, self.interval, self.period = mode, interval, period
        self.speed = min(max(float(speed), 0.2), 500.0)
        self.poll_seconds = poll_seconds
        self.market = market
        self.graph = TradingAgentsGraph(config_for(interval, **(cfg or {})), interval)
        self.status, self.message, self.error = "starting", "Preparing data...", None
        self.total_steps: int | None = None
        self.history: list[dict] = []
        self.records: list[dict] = []
        self.states: "OrderedDict[int, dict]" = OrderedDict()
        self.notes: list[str] = []
        self.started_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="agent-session")

    # ------------------------------------------------------------------ control
    def start(self) -> "TradingSession":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self.status in ("starting", "running"):
            self.status, self.message = "stopped", "Stopped by user"

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            (self._run_replay if self.mode == "replay" else self._run_live)()
        except Exception as exc:
            logger.exception("Session failed")
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
            self.message = self.error

    # ------------------------------------------------------------------ recording
    def _decide(self, row: dict, ts) -> None:
        state = self.graph.propagate(row, ts)
        with self._lock:
            idx = len(self.records)
            d = state.decision
            self.records.append({
                "i": idx, "timestamp": state.timestamp, "price": state.price, "is_decision": True,
                "rating": state.plan.rating, "conviction": state.plan.conviction,
                "smoothed": self.graph.smoothed, "action": d.action, "approved": d.approved,
                "target": d.target_exposure, "exposure": self.graph.wallet.exposure(state.price),
                "equity": self.graph.wallet.value(state.price),
                "bh_equity": self._bh(state.price), "reason": d.reasons[-1] if d.reasons else "",
                "bull": state.plan.bull_strength, "bear": state.plan.bear_strength,
                "stop_loss": self.graph.ms.stop_loss,
            })
            self.states[idx] = state.to_dict()
            while len(self.states) > KEEP_STATES:
                self.states.popitem(last=False)

    def _tick(self, price: float, ts) -> None:
        out = self.graph.mark(price, ts)
        with self._lock:
            self.records.append({
                "i": len(self.records), "timestamp": pd.Timestamp(ts).isoformat(), "price": price,
                "is_decision": False, "action": "SELL" if out["trade"] else "HOLD",
                "reason": out["trade"]["reason"] if out["trade"] else "",
                "exposure": self.graph.wallet.exposure(price), "equity": self.graph.wallet.value(price),
                "bh_equity": self._bh(price), "stop_loss": self.graph.ms.stop_loss,
            })

    def _bh(self, price: float) -> float:
        g = self.graph
        return g.cfg["initial_cash"] * (1 - g.cfg["fee"]) * price / (g.first_price or price)

    def _with_news(self, ind: pd.DataFrame) -> pd.DataFrame:
        """Attach the cached daily news scores; the news analyst abstains where there are none."""
        if "news" not in self.graph.cfg["analysts"]:
            return ind
        name, table = best_available()
        note = (f"News analyst uses the {name} news scores (up to {pd.to_datetime(table['day']).max():%Y-%m-%d})."
                if len(table) else "News analyst abstains: no news scores yet (python -m src.agents.news_sentiment).")
        if note not in self.notes:
            self.notes.append(note)
        return attach_news(ind, table)

    # ------------------------------------------------------------------ replay
    def _run_replay(self) -> None:
        bars = update.load_latest(self.interval).dropna(subset=["close"])
        start = period_start(self.period, bars.index)
        if self.period == "new" and (bars.index >= start).sum() < 2:
            start = bars.index.max() - pd.Timedelta(days=30)
            self.notes.append("No new candles synced yet - showing the last 30 days instead. "
                              "Press 'Sync now' with internet access to download them.")
        sent = sentiment_series()
        ind = attach_sentiment(compute_indicators(bars), sent["funding"], sent["fear_greed"])
        ind = self._with_news(ind)
        if "quant" in self.graph.cfg["analysts"]:
            if self.interval == "1h":
                self.notes.append(add_quant(ind, start))
            else:
                self.notes.append("Quant analyst abstains: the team's model is trained for 1h candles.")
        seg = ind.loc[ind.index >= start]
        before = len(ind) - len(seg)
        if before < WARMUP_BARS:  # keep the first 200 candles as indicator warm-up
            seg = seg.iloc[WARMUP_BARS - before:]
        if seg.empty:
            raise ValueError("no candles in the selected period")
        self.total_steps = len(seg)
        self.status = "running"
        self.message = (f"Replaying {len(seg):,} {self.interval} candles, {seg.index[0]:%Y-%m-%d %H:%M} -> "
                        f"{seg.index[-1]:%Y-%m-%d %H:%M} UTC")
        step = pd.Timedelta(self.interval)
        delay = 1.0 / self.speed
        for ts, row in zip(seg.index, seg.to_dict("records")):
            if self._stop.is_set():
                return
            self._decide(row, ts + step)  # decided when the candle closes
            time.sleep(delay)
        self.status, self.message = "finished", f"Replay finished: {len(seg):,} candles traded."

    # ------------------------------------------------------------------ live
    def _closed_candles(self) -> pd.DataFrame:
        k = self.market.fetch_klines(BINANCE_INTERVAL[self.interval], limit=600)
        now = pd.Timestamp.now(tz="UTC")
        return k[k["close_time"] < now]

    def _run_live(self) -> None:
        errors, last_closed, sent = 0, None, None
        step = pd.Timedelta(binance_mod.INTERVAL_STEP[BINANCE_INTERVAL[self.interval]])
        quant_on = "quant" in self.graph.cfg["analysts"] and self.interval == "1h"
        if "quant" in self.graph.cfg["analysts"] and not quant_on:
            self.notes.append("Quant analyst abstains: the team's model is trained for 1h candles.")
        while not self._stop.is_set():
            try:
                now = pd.Timestamp.now(tz="UTC")
                if last_closed is None or now >= last_closed + 2 * step:
                    candles = self._closed_candles()
                    newest = candles.index.max()
                    if last_closed is None or newest > last_closed:
                        if sent is None or newest.minute == 0:
                            sent = sentiment_series(live_ratio=True, market=self.market)
                        ind = attach_sentiment(compute_indicators(candles), sent["funding"],
                                               sent["fear_greed"], sent["long_short"])
                        ind = self._with_news(ind)
                        if quant_on:
                            stored = update.load_latest("1h")
                            hourly = pd.concat([stored[candles.columns], candles])
                            hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()
                            note = add_quant(ind, newest, hourly)
                            if not self.notes or self.notes[-1] != note:
                                self.notes.append(note)
                        if last_closed is None:
                            self.history = [{"timestamp": (t + step).isoformat(), "price": float(c)}
                                            for t, c in candles["close"].iloc[:-1].items()]
                        self._decide(ind.iloc[-1].to_dict(), newest + step)
                        last_closed = newest
                price = self.market.fetch_price()
                self._tick(price, pd.Timestamp.now(tz="UTC"))
                errors = 0
                self.status = "running"
                nxt = (last_closed + 2 * step) if last_closed is not None else now
                self.message = (f"Live paper trading on Binance BTCUSDT {self.interval} candles - price every "
                                f"{self.poll_seconds:.0f}s, next agent decision after {nxt:%H:%M} UTC")
            except Exception as exc:
                errors += 1
                self.message = f"Market data error ({errors}/{MAX_CONSECUTIVE_ERRORS}): {exc}"
                logger.warning(self.message)
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    self.status, self.error = "error", self.message
                    return
            self._stop.wait(self.poll_seconds)

    # ------------------------------------------------------------------ state for the dashboard
    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            records = self.records[since:]
            n = len(self.records)
            last_decision = max(self.states) if self.states else None
            latest_state = self.states[last_decision] if last_decision is not None and since <= last_decision else None
            summary = self.graph.summary(self.records[-1]["price"] if self.records else None)
        return {
            "mode": self.mode, "interval": self.interval, "period": self.period, "speed": self.speed,
            "status": self.status, "message": self.message, "error": self.error, "notes": self.notes,
            "started_at": self.started_at, "total_steps": self.total_steps, "next_index": n,
            "records": records, "summary": summary, "config": self.graph.cfg,
            "latest_state": latest_state, "latest_state_index": last_decision,
            "history": self.history if since == 0 else [],
        }

    def state_at(self, i: int) -> dict | None:
        with self._lock:
            return self.states.get(i)
