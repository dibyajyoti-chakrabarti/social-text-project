"""Feature Request Miner dashboard.

make dashboard      (uv run streamlit run apps/dashboard/app.py, offline)

Reads only precomputed pipeline outputs (data/processed, the corpus text and reports/);
the Live Analyser page loads the local SetFit model on first use.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Feature Request Miner", page_icon="🎧", layout="wide")

import loaders  # noqa: E402

loaders.require_data()

pages = [
    st.Page("views/overview.py", title="Overview", icon="📊", default=True),
    st.Page("views/explorer.py", title="Request Explorer", icon="🔎"),
    st.Page("views/trends.py", title="Trends", icon="📈"),
    st.Page("views/insights.py", title="Insights", icon="💡"),
]
st.navigation(pages).run()
