"""
Capture PNG screenshots of docs/ml/index.html (requires local HTTP — same-origin fetch for model_records).

Usage (from repo root):
  python -m http.server 8765 --directory docs   # separate terminal
  python scripts/capture_ml_dashboard_screenshots.py

Also writes docs/report-figures/ui-overlay-states.png from overlay-three-states-capture.html
(triptych: allow / warn / block). Copies Summary capture to docs/ml-health-preview-final.png.
Best-effort README previews: hub-preview-top.png, index-preview-*.png, demo-separated-tour-preview.png,
admin-preview-redesign-demo.png (if docs/admin/index.html exists).
"""
from __future__ import annotations

import functools
import http.server
import shutil
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
HUB = f"http://127.0.0.1:{PORT}/hub.html"
INDEX = f"http://127.0.0.1:{PORT}/index.html"
DEMO = f"http://127.0.0.1:{PORT}/demo/index.html"


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

    chromium_args = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=chromium_args)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.emulate_media(media="screen")

        page.goto(OVERLAY_CAPTURE, wait_until="load", timeout=60_000)
        page.wait_for_timeout(600)
        page.screenshot(path=str(FIG / "ui-overlay-states.png"), full_page=True)

        page.goto(BASE, wait_until="domcontentloaded", timeout=120_000)
        try:
            page.wait_for_load_state("networkidle", timeout=120_000)
        except Exception:
            pass
        page.wait_for_timeout(4000)
        page.screenshot(path=str(OUT / "01-summary-tab.png"), full_page=True)

        page.locator('[data-tab="pr-curve"]').click()
        page.wait_for_timeout(2800)
        page.screenshot(path=str(OUT / "02-precision-recall-tab.png"), full_page=True)

        page.locator('[data-tab="summary"]').click()
        page.wait_for_timeout(500)
        tech = page.locator("details.tech-details summary")
        if tech.count():
            tech.click()
            page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / "03-summary-technical-details-open.png"), full_page=True)

        shutil.copyfile(OUT / "01-summary-tab.png", DOCS / "ml-health-preview-final.png")

        try:
            page.goto(HUB, wait_until="domcontentloaded", timeout=120_000)
            try:
                page.wait_for_load_state("networkidle", timeout=90_000)
            except Exception:
                pass
            page.wait_for_timeout(2000)
            page.screenshot(path=str(DOCS / "hub-preview-top.png"), full_page=False)
        except Exception as ex:
            print(f"[capture] hub preview skipped: {ex}")

        try:
            page.set_viewport_size({"width": 1440, "height": 900})
            page.goto(INDEX, wait_until="domcontentloaded", timeout=120_000)
            try:
                page.wait_for_load_state("networkidle", timeout=90_000)
            except Exception:
                pass
            page.wait_for_timeout(2500)
            page.screenshot(path=str(DOCS / "index-preview-desktop-viewport.png"), full_page=True)
        except Exception as ex:
            print(f"[capture] index desktop preview skipped: {ex}")

        try:
            page.set_viewport_size({"width": 390, "height": 844})
            page.goto(INDEX, wait_until="domcontentloaded", timeout=120_000)
            try:
                page.wait_for_load_state("networkidle", timeout=90_000)
            except Exception:
                pass
            page.wait_for_timeout(1500)
            page.screenshot(path=str(DOCS / "index-preview-mobile.png"), full_page=True)
        except Exception as ex:
            print(f"[capture] index mobile preview skipped: {ex}")

        try:
            page.set_viewport_size({"width": 1440, "height": 900})
            page.goto(DEMO, wait_until="domcontentloaded", timeout=180_000)
            try:
                page.wait_for_load_state("networkidle", timeout=120_000)
            except Exception:
                pass
            page.wait_for_timeout(3500)
            page.screenshot(path=str(DOCS / "demo-separated-tour-preview.png"), full_page=True)
        except Exception as ex:
            print(f"[capture] demo preview skipped: {ex}")

        admin_html = DOCS / "admin" / "index.html"
        if admin_html.is_file():
            try:
                page.set_viewport_size({"width": 1440, "height": 900})
                page.goto(
                    f"http://127.0.0.1:{PORT}/admin/index.html",
                    wait_until="domcontentloaded",
                    timeout=120_000,
                )
                try:
                    page.wait_for_load_state("networkidle", timeout=90_000)
                except Exception:
                    pass
                page.wait_for_timeout(2000)
                page.screenshot(
                    path=str(DOCS / "admin-preview-redesign-demo.png"),
                    full_page=True,
                )
            except Exception as ex:
                print(f"[capture] admin preview skipped: {ex}")

        browser.close()

    if own_server and httpd:
        httpd.shutdown()

    print(
        f"Wrote ML screenshots under {OUT}, overlay under {FIG}, "
        f"synced ml-health-preview-final.png; README previews (hub, index, demo, admin) best-effort under {DOCS}"
    )


if __name__ == "__main__":
    main()
