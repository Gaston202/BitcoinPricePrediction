# One-command pipeline. Every step reads frozen inputs, so reruns give the same numbers.
#   make test         unit tests (offline)
#   make data         cleaned + model-ready data (notebooks 02 and 04)
#   make news         download headlines (resumable) and freeze them
#   make sentiment    lexicon scores; `make sentiment-llm` adds the LLM scores (needs OPENROUTER_API_KEY)
#   make agent        agent experiments -> reports/results/agent/
#   make notebooks    execute every notebook in order
#   make all          data -> sentiment -> agent -> notebooks -> test

PY ?= python
NEWS_SOURCE ?= hn

.PHONY: test data news sentiment sentiment-llm tune agent notebooks all

test:
	$(PY) -m pytest -q

data:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_quality.ipynb
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_feature_engineering.ipynb

news:
	$(PY) -m src.data.news --source $(NEWS_SOURCE)

sentiment:
	$(PY) -m src.agents.news_sentiment --scorer lexicon

sentiment-llm:
	$(PY) -m src.agents.news_sentiment --scorer llm

tune:
	$(PY) scripts/run_agent_evaluation.py tune

agent:
	$(PY) scripts/run_agent_evaluation.py all

notebooks:
	for nb in notebooks/0*.ipynb; do $(PY) -m jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; done

all: data sentiment agent notebooks test
