Place PNG (or PDF) exports here for the written report. Names match docs/CAPSTONE_RESULTS_CHAPTER.md (Figures 6.1–6.2, 6.3–6.8 paths).

report-figures/ (this folder)
  pr_curve.png
  roc_curve.png
  ui-overlay-states.png      — Fig 6.3 three-panel overlay (generated via scripts/capture_ml_dashboard_screenshots.py from overlay-three-states-capture.html)
  ui-layer-inspector.png     — Fig 6.4 Alt+L
  ui-landing-page.png        — Fig 6.7 docs/index.html
  ui-hub-page.png            — Fig 6.8 docs/hub.html

docs/ml/screenshots/
  01-summary-tab.png         — Fig 6.6 Summary
  02-precision-recall-tab.png
  03-summary-technical-details-open.png
  04-by-category-tab.png     — Fig 6.5 (capture category F1 section / By Category tab)

Regenerate dashboard PNGs:
  python scripts/capture_ml_dashboard_screenshots.py
(Add manual capture for 04-by-category-tab.png after scrolling to category bars or adding a By Category tab.)
