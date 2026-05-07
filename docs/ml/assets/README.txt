precision_recall_curve.png
----------------------------
Matplotlib PR figure (professor-style): blue PR line, orange min-precision target,
red “ML val-calib t” and green “ML test-rec t” markers.

Generate + publish to this folder:
  cd core-sentinel-guardrail
  python professor_final_benchmark.py
  python publish_website_records.py

publish_website_records copies outputs/precision_recall_curve.png here when present.
