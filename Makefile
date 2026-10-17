.PHONY: setup test lint format smoke scrape corpus eda data sample label gold silver detect errors

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

smoke:
	uv run python scripts/llm_smoke.py

scrape:
	uv run python scripts/scrape_playstore.py
	uv run python scripts/scrape_reddit.py

corpus:
	uv run python scripts/build_corpus.py

eda:
	uv run python scripts/eda.py

data: scrape corpus eda

sample:
	uv run python scripts/sample_gold.py

label:
	uv run streamlit run apps/labeller.py

gold:
	uv run python scripts/gold_summary.py

silver:
	uv run python scripts/silver_label.py

detect:
	uv run python scripts/evaluate_detection.py

errors:
	uv run python scripts/error_analysis.py
