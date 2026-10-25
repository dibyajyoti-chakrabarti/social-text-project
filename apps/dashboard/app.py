"""Feature Request Miner dashboard.

make dashboard      (uv run streamlit run apps/dashboard/app.py, offline)

Reads only precomputed pipeline outputs (data/processed, the corpus text and reports/);
the Live Analyser and Pipeline Trace pages load the local SetFit model on first use.
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
    st.Page("views/model_lab.py", title="Model Lab", icon="🧪"),
    st.Page("views/live.py", title="Live Analyser", icon="⚡"),
    st.Page("views/trace.py", title="Pipeline Trace", icon="🧭"),
]
st.navigation(pages).run()
