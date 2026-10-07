"""Project-wide constants. Change values here, never hardcode them elsewhere."""

from pathlib import Path

import pandas as pd

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# --- Market ------------------------------------------------------------------
SYMBOL = "BTCUSDT"
BASE_INTERVAL = "1h"  # valid both as a Binance interval and a pandas frequency
# pandas frequency strings (pandas 3 requires uppercase "D" for days).
RESAMPLE_INTERVALS = ("4h", "1D")

# Binance spot listing date for BTCUSDT; the API starts returning data here.
START_DATE = pd.Timestamp("2017-08-17 00:00", tz="UTC")
# Frozen cutoff: the last candle OPEN time included in the snapshot.
CUTOFF_DATE = pd.Timestamp("2026-09-30 23:00", tz="UTC")

# Market-data-only mirror, used if api.binance.com is geo-blocked (HTTP 451).
# Set env var BINANCE_USE_DATA_API=1 to switch.
DATA_API_URL = "https://data-api.binance.vision/api"

# --- Kline schema (order returned by /api/v3/klines) --------------------------
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
PRICE_COLUMNS = ["open", "high", "low", "close"]
VOLUME_COLUMNS = [
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

# --- Cleaning policy ---------------------------------------------------------
# Gaps of at most this many consecutive missing candles are forward-filled;
# longer gaps stay NaN and are excluded from analysis.
MAX_FFILL_GAP_HOURS = 3

# --- Labeling ----------------------------------------------------------------
# Candidate horizons and "flat" thresholds explored in EDA (item 11).
CANDIDATE_HORIZONS = ("4h", "1D")
CANDIDATE_FLAT_THRESHOLDS = (0.001, 0.0025, 0.005, 0.01)

# --- Backtest assumptions ----------------------------------------------------
TRADING_FEE = 0.001  # 0.1% per side, Binance spot default tier
