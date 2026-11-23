"""News collection, sentiment scoring and the news analyst. No network: every API is faked."""

import json

import numpy as np
import pandas as pd
import pytest

from src.agents import news_sentiment as ns
from src.agents.analysts import news_analyst
from src.agents.market_data import attach_news
from src.data import news

DAY = pd.Timestamp("2024-03-05", tz="UTC")


# --------------------------------------------------------------------------- collection
def test_clean_title_restores_gdelt_punctuation():
    assert news.clean_title("Bitcoin rises , analysts say ( again ) .") == "Bitcoin rises, analysts say (again)."


def test_tidy_day_filters_irrelevant_and_duplicate_headlines():
    raw = [
        {"seendate": "20240305T120000Z", "title": "Bitcoin hits record high", "domain": "a.com", "url": "u1"},
        {"seendate": "20240305T090000Z", "title": "Bitcoin hits record high !", "domain": "b.com", "url": "u2"},
        {"seendate": "20240305T100000Z", "title": "Stock Market", "domain": "c.com", "url": "u3"},
        {"seendate": "20240305T110000Z", "title": "Crypto lender files for bankruptcy", "domain": "d.com", "url": "u4"},
    ]
    df = news.tidy_day(raw, DAY, "gdelt")
    assert list(df["title"]) == ["Bitcoin hits record high!", "Crypto lender files for bankruptcy"]
    assert df["domain"].iloc[0] == "b.com"          # the first sighting of the duplicate is kept
    assert (df["day"] == DAY).all() and list(df["rank"]) == [0, 1]


def test_tidy_hn_ranks_by_points():
    hits = [{"title": "Bitcoin ETF approved", "url": "https://www.sec.gov/x", "created_at": "2024-03-05T10:00:00Z", "points": 5},
            {"title": "Bitcoin crashes", "url": None, "created_at": "2024-03-05T11:00:00Z", "points": 50}]
    df = news.tidy_day(hits, DAY, "hn")
    assert list(df["title"]) == ["Bitcoin crashes", "Bitcoin ETF approved"]
    assert list(df["domain"]) == ["news.ycombinator.com", "sec.gov"]


def test_download_is_resumable_and_snapshot_merges_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "NEWS_DIR", tmp_path)
    calls = []

    def fake_fetch(day, session):
        calls.append(day)
        return [{"title": f"Bitcoin news {day:%d}", "url": "", "created_at": day.isoformat(), "points": 1}]

    assert news.download("2024-03-01", "2024-03-03", "hn", fetch=fake_fetch, gap=0) == 3
    assert news.download("2024-03-01", "2024-03-04", "hn", fetch=fake_fetch, gap=0) == 1  # only the new day
    assert len(calls) == 4
    df = news.build_snapshot("hn", out=tmp_path / "h.parquet")
    assert df["day"].nunique() == 4 and (df["source"] == "hn").all()
    cov = news.coverage(df, "2024-03-01", "2024-03-06")
    assert list(cov["headlines"]) == [1, 1, 1, 1, 0, 0]


def test_download_retries_after_a_throttled_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "NEWS_DIR", tmp_path)
    monkeypatch.setattr(news.time, "sleep", lambda s: None)
    answers = iter([news.NewsError("busy"), []])

    def flaky(day, session):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    assert news.download("2024-03-01", "2024-03-01", "gdelt", fetch=flaky, gap=0) == 1


# --------------------------------------------------------------------------- scoring
def test_lexicon_scores_direction_and_negation():
    assert ns.lexicon_headline_score("Bitcoin surges to record high") > 0.5
    assert ns.lexicon_headline_score("Exchange hacked, bitcoin plunges") < -0.5
    assert ns.lexicon_headline_score("SEC does not approve bitcoin ETF") < 0
    assert ns.lexicon_headline_score("Bitcoin conference starts in Miami") is None


def test_lexicon_day_without_sentiment_words_has_zero_confidence():
    rows = ns.LexiconScorer().score_days({DAY: ["Bitcoin conference starts"]})
    assert rows[0]["confidence"] == 0 and rows[0]["score"] == 0


def test_parse_llm_json_validates_clips_and_fills_missing_days():
    text = '```json\n{"days": [{"day": 1, "score": 3, "confidence": 0.9, "reason": "ETF"}, {"day": "x"}]}\n```'
    rows = ns.parse_llm_json(text, 2)
    assert rows[0] == {"score": 1.0, "confidence": 0.9, "reason": "ETF"}
    assert rows[1]["confidence"] == 0.0


class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text or json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.payloads = list(responses), []

    def post(self, url, json=None, timeout=None, headers=None):
        self.payloads.append(json)
        return self.responses.pop(0)


def _llm_answer(n):
    content = json.dumps({"days": [{"day": k, "score": 0.5, "confidence": 0.8, "reason": "r"} for k in range(1, n + 1)]})
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def test_llm_prompt_hides_the_dates_and_parses_the_answer():
    session = FakeSession([_llm_answer(2)])
    scorer = ns.LLMScorer(api_key="test", session=session, pause=0)
    days = {DAY: ["Bitcoin rallies"], DAY + pd.Timedelta("1D"): ["Bitcoin falls"]}
    rows = scorer.score_days(days)
    prompt = session.payloads[0]["messages"][1]["content"]
    assert "2024" not in prompt and "Day 1:" in prompt and "Day 2:" in prompt
    assert session.payloads[0]["temperature"] == 0 and session.payloads[0]["model"] == ns.DEFAULT_MODEL
    assert [r["day"] for r in rows] == list(days) and rows[0]["score"] == 0.5


def test_llm_scorer_needs_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ns, "load_api_key", lambda: None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        ns.LLMScorer()


def test_load_api_key_reads_a_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('OTHER=1\nOPENROUTER_API_KEY="sk-or-abc"\n')
    assert ns.load_api_key(env) == "sk-or-abc"


def test_score_all_resumes_and_stops_cleanly_on_quota(tmp_path):
    days = pd.date_range("2024-03-01", periods=5, freq="1D", tz="UTC")
    heads = pd.DataFrame({"day": days, "rank": 0, "title": ["Bitcoin rallies"] * 5})
    session = FakeSession([_llm_answer(2), FakeResponse(429, text="Rate limit exceeded: free-models-per-day")])
    scorer = ns.LLMScorer(api_key="k", session=session, pause=0)
    path = tmp_path / "s.parquet"
    out = ns.score_all(heads, scorer, batch_days=2, path=path)
    assert len(out) == 2                                     # quota hit on the second batch
    session.responses = [_llm_answer(2), _llm_answer(1)]
    out = ns.score_all(heads, scorer, batch_days=2, path=path)
    assert len(out) == 5 and out["day"].is_unique            # resumed, nothing scored twice


# --------------------------------------------------------------------------- join + analyst
def _sentiment(scores):
    days = pd.date_range("2024-03-01", periods=len(scores), freq="1D", tz="UTC")
    return pd.DataFrame({"day": days, "score": scores, "confidence": 0.8, "n_headlines": 10,
                         "reason": "r", "scorer": "test"})


def test_attach_news_uses_a_day_only_after_it_has_ended():
    idx = pd.date_range("2024-03-01", "2024-03-04", freq="4h", tz="UTC")
    ind = pd.DataFrame({"close": 1.0}, index=idx)
    out = attach_news(ind, _sentiment([0.9, -0.9, 0.0]))
    assert out.loc["2024-03-01", "news_score"].isna().all()                  # day 1 not finished yet
    assert (out.loc["2024-03-02", "news_score"] == 0.9).all()                # day 1 known from day 2
    assert (out.loc["2024-03-03", "news_score"] == -0.9).all()
    assert out.loc["2024-03-03 00:00", "news_score_3d"] == pytest.approx(0.0)


def test_attach_news_drops_stale_scores_after_a_news_outage():
    idx = pd.date_range("2024-03-01", "2024-03-10", freq="1D", tz="UTC")
    out = attach_news(pd.DataFrame({"close": 1.0}, index=idx), _sentiment([0.5]))
    assert out.loc["2024-03-02", "news_score"] == 0.5
    assert np.isnan(out.loc["2024-03-08", "news_score"])


def test_news_analyst_votes_abstains_and_ignores_neutral_news():
    bull = news_analyst({"news_score_3d": 0.6, "news_confidence": 0.7, "news_reason": "ETF approved"})
    assert bull.score > 0 and bull.confidence == pytest.approx(0.7) and "ETF approved" in bull.evidence[0].text
    assert news_analyst({}).confidence == 0
    assert news_analyst({"news_score_3d": 0.05, "news_confidence": 0.9}).evidence == []
