"""End-to-end: model training, replay session and the dashboard API (needs data/interim)."""

import time

import pytest

from src import config

pytestmark = pytest.mark.skipif(
    not (config.INTERIM_DIR / "btcusdt_1D_clean.parquet").exists(),
    reason="run notebooks/02_data_quality.ipynb first",
)


def test_replay_session_trades_the_whole_test_window():
    from src.agent.session import TradingSession

    session = TradingSession(mode="replay", interval="1D", speed=500).start()
    session.join(timeout=120)
    state = session.snapshot()
    assert state["status"] == "finished", state["message"]
    assert state["next_index"] == state["total_steps"] > 300
    first = state["records"][0]
    assert first["forecast_for"] > first["timestamp"]  # forecast points one candle ahead
    assert state["summary"]["equity"] > 0


def test_live_session_with_a_fake_exchange():
    import pandas as pd

    from src.agent import model_service
    from src.agent.session import TradingSession

    clean = model_service.load_clean("1h").dropna().iloc[-120:].copy()
    clean.index = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=len(clean), freq="h")
    clean["close_time"] = clean.index + pd.Timedelta("1h")

    from src.agent import market

    class FakeMarket:
        provisional_frame = staticmethod(market.provisional_frame)

        @staticmethod
        def fetch_klines(interval, limit=120):
            return clean

        @staticmethod
        def fetch_price():
            return float(clean["close"].iloc[-1]) * 1.001

    session = TradingSession(mode="live", poll_seconds=0.05, decision_seconds=0.2, market=FakeMarket).start()
    deadline = time.time() + 180
    while time.time() < deadline and session.snapshot()["next_index"] < 6:
        time.sleep(0.1)
    session.stop()
    state = session.snapshot()
    assert state["error"] is None, state["message"]
    assert state["next_index"] >= 6
    assert len(state["history"]) == 119
    assert any(r["is_decision"] for r in state["records"])
    assert any(not r["is_decision"] for r in state["records"])


def test_dashboard_api_smoke():
    from fastapi.testclient import TestClient

    from dashboard.app import app

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        deadline = time.time() + 120
        while client.get("/api/overview").json()["status"] == "loading" and time.time() < deadline:
            time.sleep(0.5)
        ov = client.get("/api/overview").json()
        assert ov["status"] == "ready", ov
        assert len(ov["price"]) > 3000 and ov["figures"]
        assert client.get(f"/figures/{ov['figures'][0]['file']}").status_code == 200

        r = client.post("/api/session/start", json={"mode": "replay", "interval": "1D", "speed": 500})
        assert r.status_code == 200
        time.sleep(1.5)
        st = client.get("/api/session/state?since=0").json()
        assert st["next_index"] > 0 and st["records"]
        assert client.post("/api/session/stop").status_code == 200
        assert client.post("/api/session/start", json={"mode": "replay", "speed": -1}).status_code == 422
