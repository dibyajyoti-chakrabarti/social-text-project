"""Ping every configured model profile once to check keys, model ids and latency.

uv run python scripts/llm_smoke.py
"""

from frminer.config import load_config
from frminer.llm import LLM

PROMPT = (
    "Classify this app review as one of: feature_request, bug_report, praise. "
    'Reply with JSON {"label": ...}.\n\nReview: "would kill for a sleep timer on podcasts 💀"'
)


def main() -> None:
    for profile in load_config()["models"]:
        try:
            llm = LLM(profile)
            data, resp = llm.complete_json(PROMPT)
            status = "cached" if resp.cached else f"{resp.latency_s:.1f}s"
            today = llm.requests_today()
            name = f"{llm.provider}/{llm.model}"
            print(f"✓ {profile:<10} {name}: {data} ({status}, {today} req today)")
        except Exception as e:  # report every profile, even if one fails
            print(f"✗ {profile:<10} {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
