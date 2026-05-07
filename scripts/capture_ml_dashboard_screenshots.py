"""
Capture PNG screenshots of docs/ml/index.html (requires local HTTP — same-origin fetch for model_records).

Usage (from repo root):
  python -m http.server 8765 --directory docs   # separate terminal
  python scripts/capture_ml_dashboard_screenshots.py

Also writes docs/report-figures/ui-overlay-states.png from overlay-three-states-capture.html
(triptych: allow / warn / block).
"""
from __future__ import annotations

import functools
import http.server
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
OUT = REPO / "docs" / "ml" / "screenshots"
FIG = REPO / "docs" / "report-figures"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}/ml/index.html"
OVERLAY_CAPTURE = f"http://127.0.0.1:{PORT}/report-figures/overlay-three-states-capture.html"


def try_start_server() -> tuple[socketserver.TCPServer, threading.Thread] | tuple[None, None]:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(DOCS.resolve()),
    )

    class _ReuseAddr(socketserver.TCPServer):
        allow_reuse_address = True

    try:
        httpd = _ReuseAddr(("", PORT), handler)
    except OSError:
        return None, None

    def run():
        httpd.serve_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return httpd, t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    httpd, _ = try_start_server()
    own_server = httpd is not None
    if not own_server:
        print(f"Using existing server on port {PORT} (or failing if none).")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.emulate_media(media="screen")

        page.goto(OVERLAY_CAPTURE, wait_until="load", timeout=60_000)
        page.wait_for_timeout(600)
        page.screenshot(path=str(FIG / "ui-overlay-states.png"), full_page=True)

        page.goto(BASE, wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(4000)
        page.screenshot(path=str(OUT / "01-summary-tab.png"), full_page=True)

        page.locator('[data-tab="pr-curve"]').click()
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "02-precision-recall-tab.png"), full_page=True)

        page.locator('[data-tab="summary"]').click()
        page.wait_for_timeout(500)
        tech = page.locator("details.tech-details summary")
        if tech.count():
            tech.click()
            page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / "03-summary-technical-details-open.png"), full_page=True)

        browser.close()

    if own_server and httpd:
        httpd.shutdown()

    print(f"Wrote screenshots under {OUT} and {FIG / 'ui-overlay-states.png'}")


if __name__ == "__main__":
    main()
