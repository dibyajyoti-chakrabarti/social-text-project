"""The report and demo templates render from reports/tables with every field resolved."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_report  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    return build_report.context(ROOT / "reports" / "tables")


@pytest.mark.parametrize("template", list(build_report.TEMPLATES))
def test_template_renders(ctx, template):
    out = build_report.render((ROOT / "reports" / template).read_text(), ctx)
    assert "{{" not in out and "}}" not in out
    assert "nan" not in out.lower().split()


def test_md_formats_numbers():
    import pandas as pd

    table = build_report.md(
        pd.DataFrame({"a": [1234], "b": [0.12345], "c": [0.5]}), pct_cols=("c",)
    )
    assert "1,234" in table and "0.123" in table and "50.0%" in table
