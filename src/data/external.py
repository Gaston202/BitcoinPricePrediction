"""Frozen snapshots of the non-price inputs, cut at the same date as the price snapshot.

The live dashboard keeps syncing these series into data/live/. The evaluation must not depend
on when it is run, so it reads these frozen copies instead (committed under data/external/):

  btcusdt_funding.parquet   Binance USDT-M funding rate, every 8h since 2019-09
  fear_greed.parquet        alternative.me Crypto Fear & Greed index, daily since 2018-02
  news_headlines.parquet    Bitcoin headlines, one source per snapshot (src/data/news.py)
  news_sentiment_*.parquet  daily news scores (src/agents/news_sentiment.py)

    python -m src.data.external      # re-download funding + Fear & Greed and freeze them
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data import binance

EXTERNAL_DIR = config.DATA_DIR / "external"
FUNDING_FILE = EXTERNAL_DIR / f"{config.SYMBOL.lower()}_funding.parquet"
FEAR_GREED_FILE = EXTERNAL_DIR / "fear_greed.parquet"
FUNDING_START = pd.Timestamp("2019-09-01", tz="UTC")
# The last candle opens at CUTOFF_DATE and closes one hour later.
SNAPSHOT_END = config.CUTOFF_DATE + pd.Timedelta(config.BASE_INTERVAL)


def freeze() -> dict[str, int]:
    """Download both series and keep only values published before the snapshot end."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    funding = binance.fetch_funding_history(FUNDING_START)
    fear_greed = binance.fetch_fear_greed()
    funding[funding.index < SNAPSHOT_END].to_parquet(FUNDING_FILE)
    fear_greed[fear_greed.index < SNAPSHOT_END].to_parquet(FEAR_GREED_FILE)
    return {"funding_rows": int((funding.index < SNAPSHOT_END).sum()),
            "fear_greed_rows": int((fear_greed.index < SNAPSHOT_END).sum())}


def load_funding() -> pd.Series:
    return pd.read_parquet(FUNDING_FILE)["funding_rate"] if FUNDING_FILE.exists() else pd.Series(dtype=float)


def load_fear_greed() -> pd.Series:
    return pd.read_parquet(FEAR_GREED_FILE)["fear_greed"] if FEAR_GREED_FILE.exists() else pd.Series(dtype=float)


if __name__ == "__main__":
    print(freeze())
