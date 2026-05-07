PNG exports of the Model Health dashboard (local Playwright run).

Files
  01-summary-tab.png — Summary tab after model_records load (full page).
  02-precision-recall-tab.png — Precision–Recall tab including chart, legend, notes, sidebar (full page).
  03-summary-technical-details-open.png — Summary tab with Technical Details expanded.
  04-by-category-tab.png — Optional: per-category F1 bars (Fig 6.5 in CAPSTONE_RESULTS_CHAPTER). Capture manually if not added by the script (scroll Summary to category section or use a By Category tab when present).

Regenerate (repo root, Playwright + Chromium installed):
  python scripts/capture_ml_dashboard_screenshots.py

Also refreshes README/marketing PNGs in docs/: hub-preview-top.png, index-preview-desktop-viewport.png,
index-preview-mobile.png, demo-separated-tour-preview.png, admin-preview-redesign-demo.png (if admin exists),
and copies 01-summary-tab.png → docs/ml-health-preview-final.png.

The script serves docs/ on port 8765 if that port is free.
