"""End-to-end: replay and live sessions and the dashboard API (needs the raw snapshot in data/raw)."""

import os
import time

import pandas as pd
import pytest

from src import config

pytestmark = pytest.mark.skipif(not any(config.RAW_DIR.glob("*.parquet")), reason="no raw snapshot")
os.environ["BTC_SYNC"] = "0"  # never hit the network from the test suite


def test_replay_session_runs_the_whole_period():
    from src.agents.session import TradingSession

    s = TradingSession(mode="replay", interval="1D", period="365d", speed=500,
                       cfg={"analysts": ["trend", "momentum", "volatility", "volume"]}).start()
    s.join(timeout=120)
    st = s.snapshot()
    assert st["status"] == "finished", st["message"]
    assert st["next_index"] == st["total_steps"] > 300
    assert st["latest_state"]["plan"]["debate"] and st["latest_state"]["risk_views"]
    assert s.state_at(0)["decision"]["action"] in ("BUY", "SELL", "HOLD")


def test_live_session_with_a_fake_exchange():
    from src.agents.session import TradingSession
    from src.data import update

    h = update.load_latest("1h").dropna(subset=["close"]).iloc[-600:].copy()
    cols = [c for c in h.columns if c not in ("is_missing", "filled")]
    now = pd.Timestamp.now(tz="UTC").floor("min")
    h = h[cols]
    h.index = pd.date_range(end=now, periods=len(h), freq="1min", tz="UTC")
    h["close_time"] = h.index + pd.Timedelta("1min") - pd.Timedelta("1ms")

    class FakeExchange:
        @staticmethod
        def fetch_klines(interval, limit=600):
            return h

        @staticmethod
        def fetch_price():
            return float(h["close"].iloc[-1])

        @staticmethod
        def fetch_long_short_ratio():
            raise RuntimeError("not in tests")

    s = TradingSession(mode="live", interval="1m", poll_seconds=0.05, market=FakeExchange,
                       cfg={"analysts": ["trend", "momentum", "volume"]}).start()
    deadline = time.time() + 60
    while time.time() < deadline and s.snapshot()["next_index"] < 5:
        time.sleep(0.1)
    s.stop()
    st = s.snapshot()
    assert st["error"] is None, st["message"]
    assert any(r["is_decision"] for r in st["records"]) and any(not r["is_decision"] for r in st["records"])
    assert len(st["history"]) == len(h) - 2  # closed candles before the decision candle


def test_dashboard_api_smoke():
    from fastapi.testclient import TestClient

    from dashboard.app import app

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        deadline = time.time() + 150
        while client.get("/api/overview").json()["status"] == "loading" and time.time() < deadline:
            time.sleep(0.5)
        ov = client.get("/api/overview").json()
        assert ov["status"] == "ready", ov
        assert ov["backtests"] and ov["agent_now"]["plan"]["rating"] and len(ov["price"]) > 3000
        assert client.get(f"/figures/{ov['figures'][0]['file']}").status_code == 200
        assert "last_candle" in client.get("/api/data/status").json()

        r = client.post("/api/session/start", json={"mode": "replay", "interval": "1D", "period": "90d", "speed": 500,
                                                    "analysts": ["trend", "momentum"]})
        assert r.status_code == 200, r.text
        time.sleep(2)
        st = client.get("/api/session/state?since=0").json()
        assert st["next_index"] > 0 and st["records"]
        assert client.get("/api/session/step/0").status_code == 200
        assert client.post("/api/session/stop").status_code == 200
        bad = client.post("/api/session/start", json={"mode": "replay", "interval": "1m"})
        assert bad.status_code == 400
        assert client.post("/api/session/start", json={"analysts": []}).status_code == 422
