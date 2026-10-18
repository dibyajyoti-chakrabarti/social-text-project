"""Screenshot each dashboard page → reports/figures/dashboard_<page>.png (report §7.7).

Start the dashboard on port 8599 first, then run this with a throwaway Playwright
(not a project dependency):

    HF_HUB_OFFLINE=1 uv run streamlit run apps/dashboard/app.py --server.headless true \
        --server.port 8599 --server.address localhost
    uvx --from playwright playwright install chromium        # once
    uv run --with playwright python scripts/dashboard_screenshots.py

The Live Analyser shot runs the "Feature request" example with the LLM toggle off.
"""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8599"
OUT = Path(__file__).resolve().parents[1] / "reports" / "figures"
PAGES = ["overview", "explorer", "trends", "insights", "model_lab", "live"]


def settle(page, extra: float = 3.0) -> None:
    """Wait for the app to render and for Streamlit to stop running the script."""
    page.wait_for_selector("[data-testid='stApp']", timeout=60_000)
    for _ in range(120):
        if not page.query_selector("[data-testid='stStatusWidget']"):
            break
        time.sleep(0.5)
    time.sleep(extra)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
        for name in PAGES:
            page.goto(f"{BASE}/" if name == "overview" else f"{BASE}/{name}")
            settle(page)
            if name == "live":
                page.get_by_text("Feature request", exact=True).first.click()
                settle(page, 1)
                page.get_by_role("button", name="Analyse").click()
                page.wait_for_selector("text=Requested features", timeout=180_000)
                settle(page)
            page.screenshot(path=OUT / f"dashboard_{name}.png")
            print(f"wrote reports/figures/dashboard_{name}.png")
        browser.close()


if __name__ == "__main__":
    main()
