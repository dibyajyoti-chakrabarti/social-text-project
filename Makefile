.PHONY: setup test lint format smoke

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
