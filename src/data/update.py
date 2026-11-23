"""Keep the project's data up to date with Binance, incrementally.

The frozen raw snapshot (data/raw, up to config.CUTOFF_DATE) stays untouched so the
notebooks remain reproducible. Every candle that closed after it is appended to
data/live/btcusdt_1h_live.parquet, and the combined history is re-cleaned with the
project's own gap policy into data/interim/btcusdt_{1h,4h,1D}_latest.parquet,
which is what the trading agent and the dashboard use.

Each sync only downloads what is missing (from the last stored candle to the last
fully closed one), so it is cheap to run every few minutes.

Usage (from the project root):
    python -m src.data.update                 # sync once
    python -m src.data.update --watch         # keep syncing every 5 minutes
    python -m src.data.update --watch --every 60
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src import config
from src.data import binance
from src.data.clean import fill_gaps
from src.features.returns import resample_ohlcv

logger = logging.getLogger(__name__)

LIVE_DIR = config.DATA_DIR / "live"
LIVE_CANDLES = LIVE_DIR / f"{config.SYMBOL.lower()}_1h_live.parquet"
FUNDING_FILE = LIVE_DIR / f"{config.SYMBOL.lower()}_funding.parquet"
FEAR_GREED_FILE = LIVE_DIR / "fear_greed.parquet"
STATUS_FILE = LIVE_DIR / "status.json"
RULES = ("1h",) + tuple(config.RESAMPLE_INTERVALS)
STEP = pd.Timedelta(config.BASE_INTERVAL)
FUNDING_START = pd.Timestamp("2019-09-10", tz="UTC")  # BTCUSDT perpetual listing


def latest_path(rule: str) -> Path:
    return config.INTERIM_DIR / f"{config.SYMBOL.lower()}_{rule}_latest.parquet"


@dataclass
class SyncStatus:
    snapshot_end: str = config.CUTOFF_DATE.isoformat()
    last_candle: str | None = None
    live_candles: int = 0
    added: int = 0
    last_sync: str | None = None
    last_attempt: str | None = None
    ok: bool = True
    error: str | None = None
    sentiment: dict = field(default_factory=dict)


def read_status() -> SyncStatus:
    try:
        return SyncStatus(**json.loads(STATUS_FILE.read_text()))
    except (OSError, ValueError, TypeError):
        return SyncStatus()


def _write_status(status: SyncStatus) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(asdict(status), indent=2))


def load_snapshot() -> pd.DataFrame:
    files = sorted(config.RAW_DIR.glob(f"{config.SYMBOL.lower()}_1h_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw snapshot in {config.RAW_DIR}. Run: python -m src.data.collect")
    return pd.read_parquet(files[-1])


def load_live_candles() -> pd.DataFrame:
    return pd.read_parquet(LIVE_CANDLES) if LIVE_CANDLES.exists() else binance.klines_to_frame([])


def last_closed_open_time(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Open time of the newest hourly candle that has fully closed."""
    now = now or pd.Timestamp.now(tz="UTC")
    return now.floor("h") - STEP


def combined_raw(snapshot: pd.DataFrame | None = None, live: pd.DataFrame | None = None) -> pd.DataFrame:
    snapshot = load_snapshot() if snapshot is None else snapshot
    live = load_live_candles() if live is None else live
    raw = pd.concat([snapshot, live]) if len(live) else snapshot
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return raw[[c for c in snapshot.columns]]


def rebuild_latest(raw: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Re-clean the combined history and write the *_latest.parquet files."""
    raw = combined_raw() if raw is None else raw
    frames = {"1h": fill_gaps(raw)}
    for rule in config.RESAMPLE_INTERVALS:
        frames[rule] = resample_ohlcv(frames["1h"], rule)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for rule, df in frames.items():
        df.to_parquet(latest_path(rule))
    return frames


def load_latest(rule: str = "1h") -> pd.DataFrame:
    """Cleaned candles including every synced update (built from the snapshot if never synced)."""
    path = latest_path(rule)
    if not path.exists():
        return rebuild_latest()[rule]
    return pd.read_parquet(path)


def _sync_sentiment(status: SyncStatus) -> None:
    """Funding-rate history (Binance futures) and the Fear & Greed index. Optional: failures
    are recorded but never stop the candle sync."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        old = pd.read_parquet(FUNDING_FILE) if FUNDING_FILE.exists() else None
        start = FUNDING_START if old is None or old.empty else old.index.max() + pd.Timedelta("1ms")
        new = binance.fetch_funding_history(start)
        merged = new if old is None else pd.concat([old, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_parquet(FUNDING_FILE)
        status.sentiment["funding"] = {"ok": True, "rows": len(merged),
                                       "last": merged.index.max().isoformat() if len(merged) else None}
    except Exception as exc:  # optional source
        status.sentiment["funding"] = {"ok": False, "error": str(exc)[:200]}
    try:
        fng = binance.fetch_fear_greed()
        fng.to_parquet(FEAR_GREED_FILE)
        status.sentiment["fear_greed"] = {"ok": True, "rows": len(fng), "last": fng.index.max().isoformat()}
    except Exception as exc:
        status.sentiment["fear_greed"] = {"ok": False, "error": str(exc)[:200]}


def sync(now: pd.Timestamp | None = None, with_sentiment: bool = True, fetch=None) -> SyncStatus:
    """Download every hourly candle that closed since the last stored one; rebuild latest files.

    `fetch(start, end) -> DataFrame` can be injected for tests.
    """
    fetch = fetch or (lambda s, e: binance.fetch_klines_range(s, e, "1h"))
    status = read_status()
    status.last_attempt = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    snapshot = load_snapshot()
    live = load_live_candles()
    last_stored = max(snapshot.index.max(), live.index.max() if len(live) else snapshot.index.max())
    target = last_closed_open_time(now)
    added = 0
    try:
        if target > last_stored:
            new = fetch(last_stored + STEP, target)
            new = new[(new.index > last_stored) & (new.index <= target)]
            if len(new):
                live = pd.concat([live, new[live.columns] if len(live) else new])
                live = live[~live.index.duplicated(keep="last")].sort_index()
                LIVE_DIR.mkdir(parents=True, exist_ok=True)
                live.to_parquet(LIVE_CANDLES)
                added = len(new)
        status.ok, status.error = True, None
        status.last_sync = status.last_attempt
    except Exception as exc:
        status.ok, status.error = False, str(exc)[:300]
        logger.warning("Candle sync failed: %s", exc)
    if added or not latest_path("1h").exists():
        rebuild_latest(combined_raw(snapshot, live))
    if with_sentiment and status.ok:
        _sync_sentiment(status)
    combined_last = max(snapshot.index.max(), live.index.max() if len(live) else snapshot.index.max())
    status.last_candle = combined_last.isoformat()
    status.live_candles = int(len(live))
    status.added = added
    _write_status(status)
    return status


def load_funding() -> pd.Series:
    if not FUNDING_FILE.exists():
        return pd.Series(dtype=float, name="funding_rate")
    return pd.read_parquet(FUNDING_FILE)["funding_rate"]


def load_fear_greed() -> pd.Series:
    if not FEAR_GREED_FILE.exists():
        return pd.Series(dtype=float, name="fear_greed")
    return pd.read_parquet(FEAR_GREED_FILE)["fear_greed"]


class BackgroundUpdater:
    """Thread that calls sync() every `every` seconds; listeners run after new candles arrive."""

    def __init__(self, every: float = 300.0) -> None:
        self.every = every
        self.status = read_status()
        self.running = False
        self._listeners: list = []
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="data-updater")

    def on_new_data(self, fn) -> None:
        self._listeners.append(fn)

    def start(self) -> "BackgroundUpdater":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def sync_now(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.running = True
            try:
                self.status = sync()
                if self.status.added:
                    for fn in self._listeners:
                        fn(self.status)
            except Exception as exc:
                logger.exception("Data update failed")
                self.status.ok, self.status.error = False, str(exc)[:300]
            self.running = False
            self._wake.wait(self.every)
            self._wake.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", action="store_true", help="keep syncing")
    parser.add_argument("--every", type=float, default=300, help="seconds between syncs with --watch")
    parser.add_argument("--no-sentiment", action="store_true", help="skip funding rate / fear & greed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    while True:
        st = sync(with_sentiment=not args.no_sentiment)
        msg = (f"data up to {st.last_candle} | +{st.added} new candles | "
               f"{st.live_candles} candles since the snapshot")
        print(msg if st.ok else f"sync failed: {st.error}")
        if not args.watch:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
