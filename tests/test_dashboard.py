"""Smoke test: every dashboard page renders without an exception.

Needs the data bundle (data/processed, data/interim/corpus.parquet), so it is skipped in CI.
The Live Analyser is only opened, not run, so no model is loaded.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "dashboard" / "app.py"
NEEDED = [
    ROOT / "data" / "processed" / "theme_stats.parquet",
    ROOT / "data" / "interim" / "corpus.parquet",
]
PAGES = [
    "views/overview.py",
    "views/explorer.py",
    "views/trends.py",
    "views/insights.py",
    "views/model_lab.py",
    "views/live.py",
]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in NEEDED), reason="data bundle not unpacked"
)


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    app.switch_page(page)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.title, "page has no title"
