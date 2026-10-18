.PHONY: setup test lint format smoke scrape corpus eda data sample label gold silver detect errors classify extract extract-eval cluster purity dashboard report

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

classify:
	uv run python scripts/classify_corpus.py

extract-eval:
	uv run python scripts/evaluate_extraction.py

extract:
	uv run python scripts/extract_corpus.py

cluster:
	uv run python scripts/cluster_requests.py

purity:
	uv run streamlit run apps/purity.py

# Offline demo: Hugging Face models must already be in the local cache (they are after setup runs).
dashboard:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run streamlit run apps/dashboard/app.py

# Report figures + render reports/report.md and reports/demo_script.md from their templates.
report:
	uv run python scripts/report_figures.py
	uv run python scripts/build_report.py
