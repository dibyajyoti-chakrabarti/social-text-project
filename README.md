# Feature Request Miner

A text analytics dashboard for extracting customer feature requests from unstructured product feedback (Spotify reviews and Reddit discussions).

Advanced Social Text project. See [PLAN.md](PLAN.md) for the full plan; **§0 Handoff** there has the current status, setup (including the data bundle, which is not in git) and next steps.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
make setup               # install Python 3.12 deps into .venv
cp .env.example .env     # then fill in API keys (GEMINI_KEY_1.., GROQ_KEY, OPENROUTER_API_KEY; all free tier)
make test                # unit tests (no API calls)
make smoke               # one live call per configured LLM profile
```

LLM responses are cached in `data/cache/llm.sqlite`, so re-running any stage costs no quota.

## Dashboard

```bash
make dashboard           # http://localhost:8501, runs offline (needs the data bundle, see PLAN.md §0)
```

Six pages: Overview, Request Explorer, Trends, Insights, Model Lab, Live Analyser. The Live Analyser classifies pasted text with the local SetFit model; its LLM extraction toggle is off by default.

## Annotation

```bash
make label               # opens the labelling app in your browser
```

Keys: `1`–`6` choose a label · `u` unsure · `b`/`←` back · `n`/`→` next. Labels are saved after every item to `annotations/gold_labels.csv`, so you can stop at any time. See `annotations/guidelines.md`.
