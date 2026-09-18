"""Capture the dashboard screenshots used in the README.

Both servers must already be running:

    uvicorn app.main:app --app-dir backend --port 8000
    cd frontend && npm run build && npm start

Then:  python scripts/capture_screenshots.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "images"
WEB = "http://127.0.0.1:3000"
API = "http://127.0.0.1:8000"

VIEWPORT = {"width": 1440, "height": 900}


def _get(path: str):
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as resp:
        return json.loads(resp.read())


def pick_interesting_claim() -> str:
    """A claim with a decision, several risk signals and graph neighbours."""
    investigations = _get("/api/investigations?limit=60")
    scored = [
        inv
        for inv in investigations
        if inv.get("decision") and (inv.get("risk_signals") or [])
    ]
    scored.sort(key=lambda i: -(len(i.get("risk_signals") or [])))
    for inv in scored:
        graph = _get(f"/api/graph/claims/{inv['claim_id']}?depth=2")
        if len(graph.get("nodes", [])) >= 3:
            return inv["claim_id"]
    return scored[0]["claim_id"] if scored else investigations[0]["claim_id"]


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    claim_id = pick_interesting_claim()
    print(f"Featured claim: {claim_id}")

    # full_page=False for the long list views: a full-page capture of a
    # 100-row table is several megabytes and unreadable in a README.
    pages = [
        ("dashboard", "/", True),
        ("claims", "/claims", False),
        ("claim-detail", f"/claims/{claim_id}", True),
        ("fraud-rings", "/rings", False),
        ("pipeline", "/pipeline", True),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--force-color-profile=srgb"])
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = context.new_page()
        for name, path, full in pages:
            page.goto(f"{WEB}{path}", wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(900)
            target = OUT_DIR / f"{name}.png"
            page.screenshot(path=str(target), full_page=full)
            print(f"  {target.relative_to(REPO_ROOT)}")
        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Screenshot capture failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
