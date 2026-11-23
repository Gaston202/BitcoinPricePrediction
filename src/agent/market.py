"""Public Binance market data for the live paper-trading agent (no API key needed).

Tries api.binance.com first and falls back to the market-data mirror
(data-api.binance.vision), which also works where the main API returns
HTTP 451 for the user's location.
"""

from __future__ import annotations

import pandas as pd
import requests

from src import config

BASE_URLS = ("https://api.binance.com/api/v3", f"{config.DATA_API_URL}/v3")
TIMEOUT = 8


class MarketDataError(RuntimeError):
    pass


def _get(path: str, params: dict) -> object:
    errors = []
    for base in BASE_URLS:
        try:
            resp = requests.get(f"{base}/{path}", params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            errors.append(f"{base.split('/')[2]} ({status or type(exc).__name__})")
    raise MarketDataError("Could not reach Binance: " + ", ".join(errors)
                          + ". Check your internet connection or firewall.")


def fetch_price(symbol: str = config.SYMBOL) -> float:
    """Latest traded price."""
    return float(_get("ticker/price", {"symbol": symbol})["price"])


def klines_to_frame(rows: list[list]) -> pd.DataFrame:
    """Same schema as src.data.collect.klines_to_frame, without the python-binance import."""
    df = pd.DataFrame(rows, columns=config.KLINE_COLUMNS).drop(columns="ignore")
    for col in ("open_time", "close_time"):
        df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    float_cols = config.PRICE_COLUMNS + [c for c in config.VOLUME_COLUMNS if c != "number_of_trades"]
    df[float_cols] = df[float_cols].astype("float64")
    df["number_of_trades"] = df["number_of_trades"].astype("int64")
    return df.set_index("open_time").sort_index()


def fetch_klines(interval: str = "1h", limit: int = 100, symbol: str = config.SYMBOL) -> pd.DataFrame:
    """Most recent `limit` candles; the last one is the still-open candle."""
    return klines_to_frame(_get("klines", {"symbol": symbol, "interval": interval, "limit": limit}))


def provisional_frame(klines: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Make the still-open last candle usable as a feature row.

    Its volume so far is extrapolated to a full candle, so `volume_z` does not
    read every fresh candle as an abnormally quiet one.
    """
    df = klines.copy()
    now = now or pd.Timestamp.now(tz="UTC")
    start = df.index[-1]
    length = df["close_time"].iloc[-1] - start
    elapsed = min(max((now - start) / length, 0.05), 1.0)
    for col in ("volume", "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"):
        df.iloc[-1, df.columns.get_loc(col)] = df[col].iloc[-1] / elapsed
    return df
