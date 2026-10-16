# Feature Request Miner

A text analytics dashboard for extracting customer feature requests from unstructured product feedback (Spotify reviews and Reddit discussions).

Advanced Social Text project. See [PLAN.md](PLAN.md) for the full project plan.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
make setup               # install Python 3.12 deps into .venv
cp .env.example .env     # then fill in API keys (Gemini, Groq, OpenRouter; all free tier)
make test                # unit tests (no API calls)
make smoke               # one live call per configured LLM profile
```

LLM responses are cached in `data/cache/llm.sqlite`, so re-running any stage costs no quota.

## Annotation

```bash
make label               # opens the labelling app in your browser
```

Keys: `1`–`6` choose a label · `u` unsure · `b`/`←` back · `n`/`→` next. Labels are saved after every item to `annotations/gold_labels.csv`, so you can stop at any time. See `annotations/guidelines.md`.
