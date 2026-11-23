"""Collect BTCUSDT klines from the Binance REST API into an immutable raw snapshot.

Usage (from project root):
    python -m src.data.collect                 # full history up to CUTOFF_DATE
    python -m src.data.collect --end now       # up to the latest closed candle
    python -m src.data.collect --force         # overwrite an existing snapshot

No API key is needed: klines are public market data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from requests.exceptions import RequestException

from src import config

logger = logging.getLogger(__name__)

PAGE_LIMIT = 1000  # Binance max candles per request
MAX_RETRIES = 5
BACKOFF_SECONDS = 2.0
# HTTP codes worth retrying: rate limited (429), IP auto-banned (418), server errors.
RETRYABLE_STATUS = {418, 429, 500, 502, 503, 504}
INTERVAL_MS = {"1h": 3_600_000}


def make_client() -> Client:
    """Create an unauthenticated client, optionally pointed at the data-only mirror."""
    client = Client(ping=False)
    if os.getenv("BINANCE_USE_DATA_API") == "1":
        client.API_URL = config.DATA_API_URL
        logger.info("Using market-data mirror %s", config.DATA_API_URL)
    return client


def _to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _get_page_with_retry(client: Client, symbol: str, interval: str,
                         start_ms: int, end_ms: int) -> list[list]:
    """Fetch one page of klines, retrying transient failures with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.get_klines(symbol=symbol, interval=interval, startTime=start_ms,
                                     endTime=end_ms, limit=PAGE_LIMIT)
        except BinanceAPIException as exc:
            if exc.status_code == 451:
                raise RuntimeError(
                    "Binance blocked this location (HTTP 451). "
                    "Re-run with BINANCE_USE_DATA_API=1."
                ) from exc
            if exc.status_code not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
                raise
            error = exc
        except (BinanceRequestException, RequestException) as exc:
            if attempt == MAX_RETRIES:
                raise
            error = exc
        wait = BACKOFF_SECONDS * 2 ** (attempt - 1)
        logger.warning("Attempt %d failed (%s); retrying in %.0fs", attempt, error, wait)
        time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_klines(client: Client, symbol: str, interval: str,
                 start: pd.Timestamp, end: pd.Timestamp) -> list[list]:
    """Paginate through [start, end] (candle open times, inclusive)."""
    step_ms = INTERVAL_MS[interval]
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    rows: list[list] = []
    while start_ms <= end_ms:
        page = _get_page_with_retry(client, symbol, interval, start_ms, end_ms)
        if not page:
            break
        rows.extend(page)
        start_ms = page[-1][0] + step_ms
        logger.info("Fetched %d rows, up to %s", len(rows),
                    pd.to_datetime(page[-1][0], unit="ms", utc=True))
    return rows


def klines_to_frame(rows: list[list]) -> pd.DataFrame:
    """Convert raw kline lists to a typed DataFrame indexed by UTC open time."""
    df = pd.DataFrame(rows, columns=config.KLINE_COLUMNS).drop(columns="ignore")
    for col in ("open_time", "close_time"):
        df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    float_cols = config.PRICE_COLUMNS + [c for c in config.VOLUME_COLUMNS
                                         if c != "number_of_trades"]
    df[float_cols] = df[float_cols].astype("float64")
    df["number_of_trades"] = df["number_of_trades"].astype("int64")
    return df.set_index("open_time").sort_index()


def snapshot_path(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    name = f"{symbol.lower()}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    return config.RAW_DIR / name


def save_snapshot(df: pd.DataFrame, path: Path, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Write the parquet file plus a provenance metadata JSON next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    metadata = {
        "symbol": config.SYMBOL,
        "interval": config.BASE_INTERVAL,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "first_open_time": df.index.min().isoformat(),
        "last_open_time": df.index.max().isoformat(),
        "row_count": len(df),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Binance REST /api/v3/klines via python-binance",
        "python_binance_version": version("python-binance"),
    }
    path.with_name(path.stem + "_metadata.json").write_text(json.dumps(metadata, indent=2))


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the frozen raw snapshot (default: the configured symbol/interval/cutoff)."""
    path = path or snapshot_path(config.SYMBOL, config.BASE_INTERVAL,
                                 config.START_DATE, config.CUTOFF_DATE)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data.collect")
    return pd.read_parquet(path)


def _parse_end(value: str) -> pd.Timestamp:
    if value == "now":
        # Last fully closed hourly candle.
        return pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    return pd.Timestamp(value, tz="UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--end", default=config.CUTOFF_DATE.isoformat(),
                        help="last candle open time (UTC) or 'now'")
    parser.add_argument("--force", action="store_true", help="overwrite existing snapshot")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start, end = config.START_DATE, _parse_end(args.end)
    path = snapshot_path(config.SYMBOL, config.BASE_INTERVAL, start, end)
    if path.exists() and not args.force:
        logger.info("Snapshot already exists at %s (use --force to refetch)", path)
        return

    rows = fetch_klines(make_client(), config.SYMBOL, config.BASE_INTERVAL, start, end)
    if not rows:
        raise RuntimeError("Binance returned no data; check symbol, interval and dates.")
    df = klines_to_frame(rows)  # raw stays untouched; quality issues are handled in clean.py
    save_snapshot(df, path, start, end)
    logger.info("Saved %d rows to %s", len(df), path)


if __name__ == "__main__":
    main()
