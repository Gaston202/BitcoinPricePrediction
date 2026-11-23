import pandas as pd
import pytest

from src import config
from src.data import binance, update


def _klines(start, n):
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="open_time")
    rows = [[int(t.timestamp() * 1000), "100", "101", "99", "100.5", "10", int((t + pd.Timedelta("1h")).timestamp() * 1000) - 1,
             "1005", 50, "6", "603", "0"] for t in idx]
    return binance.klines_to_frame(rows)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the updater at a temporary project with a tiny 'snapshot'."""
    raw = tmp_path / "raw"
    raw.mkdir()
    snap = _klines("2026-09-29 00:00", 48)
    snap.to_parquet(raw / "btcusdt_1h_20260929_20260930.parquet")
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path / "interim")
    live = tmp_path / "live"
    for name, value in {"LIVE_DIR": live, "LIVE_CANDLES": live / "c.parquet", "FUNDING_FILE": live / "f.parquet",
                        "FEAR_GREED_FILE": live / "g.parquet", "STATUS_FILE": live / "s.json"}.items():
        monkeypatch.setattr(update, name, value)
    return snap


def test_klines_to_frame_schema():
    df = _klines("2026-01-01", 3)
    assert list(df.columns) == [c for c in config.KLINE_COLUMNS if c not in ("open_time", "ignore")]
    assert df.index.tz is not None and df["close"].dtype == "float64"


def test_sync_downloads_only_missing_closed_candles(sandbox):
    calls = []

    def fake_fetch(start, end):
        calls.append((start, end))
        return _klines(start, int((end - start) / pd.Timedelta("1h")) + 2)  # one extra, still-open candle

    now = pd.Timestamp("2026-10-01 05:30", tz="UTC")
    st = update.sync(now=now, with_sentiment=False, fetch=fake_fetch)
    assert st.ok and st.added == 5  # 00:00 .. 04:00 closed, 05:00 still open
    assert calls == [(pd.Timestamp("2026-10-01 00:00", tz="UTC"), pd.Timestamp("2026-10-01 04:00", tz="UTC"))]
    latest = update.load_latest("1h")
    assert latest.index.max() == pd.Timestamp("2026-10-01 04:00", tz="UTC")

    st2 = update.sync(now=now, with_sentiment=False, fetch=fake_fetch)  # nothing new
    assert st2.added == 0 and len(calls) == 1
    st3 = update.sync(now=now + pd.Timedelta("2h"), with_sentiment=False, fetch=fake_fetch)
    assert st3.added == 2 and calls[-1][0] == pd.Timestamp("2026-10-01 05:00", tz="UTC")
    assert update.read_status().live_candles == 7


def test_sync_failure_keeps_stored_data(sandbox):
    def broken(start, end):
        raise binance.MarketDataError("offline")

    st = update.sync(now=pd.Timestamp("2026-10-02", tz="UTC"), with_sentiment=False, fetch=broken)
    assert not st.ok and "offline" in st.error
    assert update.load_latest("1h").index.max() == sandbox.index.max()


def test_latest_frames_are_resampled(sandbox):
    frames = update.rebuild_latest()
    assert set(frames) == {"1h", "4h", "1D"} and len(frames["4h"]) == 12
