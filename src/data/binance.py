"""Public Binance market data over plain HTTPS (no API key, no python-binance needed).

Spot klines and prices come from api.binance.com, falling back to the market-data
mirror data-api.binance.vision (works where the main API answers HTTP 451).
Derivatives sentiment (funding rate, long/short ratio) comes from the USDT-M
futures API, which has no mirror - callers must treat it as optional.
"""

from __future__ import annotations

import pandas as pd
import requests

from src import config

SPOT_URLS = ("https://api.binance.com/api/v3", f"{config.DATA_API_URL}/v3")
FUTURES_URL = "https://fapi.binance.com"
TIMEOUT = 10
PAGE_LIMIT = 1000
INTERVAL_STEP = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


class MarketDataError(RuntimeError):
    pass


def _get_json(urls: tuple[str, ...], path: str, params: dict) -> object:
    errors = []
    for base in urls:
        try:
            resp = requests.get(f"{base}/{path}", params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            errors.append(f"{base.split('/')[2]} ({status or type(exc).__name__})")
    raise MarketDataError("Could not reach Binance: " + ", ".join(errors)
                          + ". Check your internet connection or firewall.")


# --------------------------------------------------------------------------- spot
def klines_to_frame(rows: list[list]) -> pd.DataFrame:
    """Raw /klines rows -> typed frame indexed by UTC open time (same schema as the snapshot)."""
    if not rows:
        return pd.DataFrame(columns=[c for c in config.KLINE_COLUMNS if c not in ("open_time", "ignore")],
                            index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
    df = pd.DataFrame(rows, columns=config.KLINE_COLUMNS).drop(columns="ignore")
    for col in ("open_time", "close_time"):
        df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    floats = config.PRICE_COLUMNS + [c for c in config.VOLUME_COLUMNS if c != "number_of_trades"]
    df[floats] = df[floats].astype("float64")
    df["number_of_trades"] = df["number_of_trades"].astype("int64")
    return df.set_index("open_time").sort_index()


def fetch_klines(interval: str = "1h", limit: int = 500, start: pd.Timestamp | None = None,
                 end: pd.Timestamp | None = None, symbol: str = config.SYMBOL) -> pd.DataFrame:
    """One page (<= 1000) of candles. Without start/end: the most recent `limit` candles,
    the last of which is the still-open one."""
    params: dict = {"symbol": symbol, "interval": interval, "limit": min(limit, PAGE_LIMIT)}
    if start is not None:
        params["startTime"] = int(start.timestamp() * 1000)
    if end is not None:
        params["endTime"] = int(end.timestamp() * 1000)
    return klines_to_frame(_get_json(SPOT_URLS, "klines", params))


def fetch_klines_range(start: pd.Timestamp, end: pd.Timestamp, interval: str = "1h",
                       symbol: str = config.SYMBOL) -> pd.DataFrame:
    """All candles whose open time is in [start, end], paginating 1000 at a time."""
    step = pd.Timedelta(INTERVAL_STEP[interval])
    pages, cursor = [], start
    while cursor <= end:
        page = fetch_klines(interval, PAGE_LIMIT, start=cursor, end=end, symbol=symbol)
        if page.empty:
            break
        pages.append(page)
        cursor = page.index[-1] + step
    if not pages:
        return klines_to_frame([])
    out = pd.concat(pages)
    return out[~out.index.duplicated(keep="last")]


def fetch_price(symbol: str = config.SYMBOL) -> float:
    return float(_get_json(SPOT_URLS, "ticker/price", {"symbol": symbol})["price"])


# --------------------------------------------------------------------------- futures sentiment
def fetch_funding_history(start: pd.Timestamp | None = None, symbol: str = config.SYMBOL) -> pd.DataFrame:
    """8-hourly funding rates (fraction per 8h) from `start` onwards, paginated."""
    rows, cursor = [], start
    while True:
        params: dict = {"symbol": symbol, "limit": PAGE_LIMIT}
        if cursor is not None:
            params["startTime"] = int(cursor.timestamp() * 1000)
        page = _get_json((FUTURES_URL,), "fapi/v1/fundingRate", params)
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_LIMIT:
            break
        cursor = pd.to_datetime(page[-1]["fundingTime"], unit="ms", utc=True) + pd.Timedelta("1ms")
    if not rows:
        return pd.DataFrame({"funding_rate": pd.Series(dtype=float)},
                            index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"funding_rate": df["fundingRate"].astype(float).to_numpy()},
                       index=pd.DatetimeIndex(pd.to_datetime(df["fundingTime"], unit="ms", utc=True), name="time"))
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_long_short_ratio(period: str = "1h", limit: int = 500, symbol: str = config.SYMBOL) -> pd.DataFrame:
    """Global account long/short ratio (Binance keeps only ~30 days of it)."""
    rows = _get_json((FUTURES_URL,), "futures/data/globalLongShortAccountRatio",
                     {"symbol": symbol, "period": period, "limit": min(limit, 500)})
    if not rows:
        return pd.DataFrame({"long_short_ratio": pd.Series(dtype=float)},
                            index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.DataFrame(rows)
    return pd.DataFrame({"long_short_ratio": df["longShortRatio"].astype(float).to_numpy()},
                        index=pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True), name="time"))


# --------------------------------------------------------------------------- fear & greed (non-Binance)
FNG_URL = "https://api.alternative.me"


def fetch_fear_greed() -> pd.DataFrame:
    """Crypto Fear & Greed index (0 = extreme fear, 100 = extreme greed), full daily history."""
    data = _get_json((FNG_URL,), "fng/", {"limit": 0, "format": "json"})["data"]
    idx = pd.DatetimeIndex(pd.to_datetime([int(d["timestamp"]) for d in data], unit="s", utc=True), name="time")
    return pd.DataFrame({"fear_greed": [float(d["value"]) for d in data]}, index=idx).sort_index()
