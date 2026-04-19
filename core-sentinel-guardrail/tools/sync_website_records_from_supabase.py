from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parent / "docs"
OUT_PATH = DOCS / "data" / "model_records.json"
DEFAULT_TABLE = "retrain_runs"


def _to_float(v: Any) -> float | None:
    try:
        n = float(v)
        return n if n == n else None
    except (TypeError, ValueError):
        return None


def _to_json_obj(v: Any) -> dict[str, Any]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _fetch_latest_retrain_row(url: str, key: str, table: str) -> dict[str, Any]:
    qs = (
        "select=run_at,accuracy,precision,recall,f1,recall_high,recall_med,"
        "model_version_tag,fpr_non_high_as_high,summary_json"
        "&order=run_at.desc.nullslast&limit=1"
    )
    endpoint = f"{url.rstrip('/')}/rest/v1/{urllib.parse.quote(table)}?{qs}"
    req = urllib.request.Request(
        endpoint,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = resp.read().decode("utf-8", errors="replace")
    rows = json.loads(payload)
    if isinstance(rows, list) and rows:
        row = rows[0]
        return row if isinstance(row, dict) else {}
    return {}


def _merge_full_summary_for_dashboard(payload: dict[str, Any], summary: dict[str, Any]) -> None:
    """
    Supabase `summary_json` from auto-retrain already includes evaluation.* curve JSON.
    Without this merge, scheduled sync wrote a slim `model_records.json` and the hosted ML
    dashboard could not draw PR/ROC (it cannot fetch core-sentinel-guardrail/reports/ on GitHub Pages).
    """
    if not summary:
        return
    sm = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    pm = payload.setdefault("metrics", {})
    for key in (
        "roc_auc",
        "confusion_holdout",
        "metric_notes",
        "false_negative_rate_risky",
        "true_negative_rate_safe",
        "macro_f1",
        "safe_recall",
        "risky_auprc",
        "recommended_threshold",
        "selection_rule",
    ):
        if key in sm and sm[key] is not None:
            pm[key] = sm[key]

    if isinstance(summary.get("evaluation"), dict):
        payload["evaluation"] = summary["evaluation"]
    if isinstance(summary.get("charts"), dict):
        payload["charts"] = summary["charts"]
    if isinstance(summary.get("policy"), dict):
        payload["policy"] = summary["policy"]
    if isinstance(summary.get("model"), dict):
        payload["model"] = summary["model"]
    if isinstance(summary.get("feedback"), dict):
        payload["feedback"] = summary["feedback"]

    cal_out = payload.setdefault("calibration", {})
    top_cal = summary.get("calibration")
    if isinstance(top_cal, dict):
        for k, v in top_cal.items():
            if v is not None:
                cal_out[k] = v

    ev = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
    tc = ev.get("threshold_calibration") if isinstance(ev.get("threshold_calibration"), dict) else {}
    rec = tc.get("recommended") if isinstance(tc.get("recommended"), dict) else {}
    if rec:
        fill = {
            "prob_threshold_risky": _to_float(rec.get("prob_threshold_risky")),
            "accuracy": _to_float(rec.get("accuracy")),
            "precision": _to_float(rec.get("precision")),
            "recall": _to_float(rec.get("recall")),
            "f1": _to_float(rec.get("f1")),
            "fpr": _to_float(rec.get("fpr")),
        }
        for k, v in fill.items():
            if v is not None and cal_out.get(k) is None:
                cal_out[k] = v
        sr_tc = tc.get("selection_rule")
        if sr_tc and cal_out.get("selection_rule") is None:
            cal_out["selection_rule"] = sr_tc


def build_records_from_row(row: dict[str, Any]) -> dict[str, Any]:
    summary = _to_json_obj(row.get("summary_json"))
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), dict) else {}
    calibration = summary.get("calibration") if isinstance(summary.get("calibration"), dict) else {}
    release_gate = summary.get("release_gate") if isinstance(summary.get("release_gate"), dict) else {}

    acc = _to_float(metrics.get("accuracy")) or _to_float(row.get("accuracy"))
    prec = _to_float(metrics.get("precision")) or _to_float(row.get("precision"))
    rec = _to_float(metrics.get("recall")) or _to_float(row.get("recall"))
    f1 = _to_float(metrics.get("f1")) or _to_float(row.get("f1"))
    fpr = _to_float(metrics.get("fpr_non_high_as_high")) or _to_float(row.get("fpr_non_high_as_high"))

    payload = {
        "artifact_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "mode": "supabase_scheduled_sync",
            "table": os.environ.get("GUARDRAIL_RETRAIN_SUPABASE_TABLE", DEFAULT_TABLE),
            "run_at": row.get("run_at"),
            "model_version_tag": row.get("model_version_tag"),
        },
        "dataset": {
            "train_samples": int(dataset.get("train_samples", 10455)),
            "val_samples": int(dataset.get("val_samples", 485)),
            "test_samples": int(dataset.get("test_samples", 2289)),
            "total_samples_approx": int(dataset.get("total_samples_approx", 13229)),
            "label_balance_train": dataset.get(
                "label_balance_train", {"low": 3485, "med": 3485, "high": 3485}
            ),
            "pii_categories": int(dataset.get("pii_categories", 24)),
            "data_sources": int(dataset.get("data_sources", 6)),
            "source_names": dataset.get(
                "source_names",
                [
                    "NVIDIA Nemotron-PII pool",
                    "Enron email pool",
                    "Faker business synthetic",
                    "safe negatives",
                    "Patronus stub set",
                    "user feedback merges",
                ],
            ),
        },
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "macro_f1": _to_float(metrics.get("macro_f1")),
            "safe_recall": _to_float(metrics.get("safe_recall")),
            "fpr_non_high_as_high": fpr,
            "auc": _to_float(metrics.get("auc")),
            "risky_auprc": _to_float(metrics.get("risky_auprc")),
            "recommended_threshold": _to_float(metrics.get("recommended_threshold")),
            "selection_rule": metrics.get("selection_rule"),
        },
        "calibration": {
            "prob_threshold_risky": _to_float(calibration.get("prob_threshold_risky")),
            "accuracy": _to_float(calibration.get("accuracy")),
            "precision": _to_float(calibration.get("precision")),
            "recall": _to_float(calibration.get("recall")),
            "f1": _to_float(calibration.get("f1")),
            "fpr": _to_float(calibration.get("fpr")),
            "selection_rule": calibration.get("selection_rule"),
        },
        "release_gate": {
            "gate_passed": release_gate.get("gate_passed"),
            "promote_allowed": release_gate.get("promote_allowed"),
            "recommended_rollout": release_gate.get("recommended_rollout"),
        },
    }
    _merge_full_summary_for_dashboard(payload, summary)
    return payload


def main() -> None:
    url = (os.environ.get("SUPABASE_URL") or os.environ.get("GUARDRAIL_SUPABASE_URL") or "").strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("GUARDRAIL_SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    table = (os.environ.get("GUARDRAIL_RETRAIN_SUPABASE_TABLE") or DEFAULT_TABLE).strip()

    if not url or not key:
        raise SystemExit("SUPABASE_URL and a Supabase key are required.")

    row = _fetch_latest_retrain_row(url, key, table)
    if not row:
        msg = (
            f"No retrain row found in Supabase table {table!r}. "
            "Publish at least one row (auto-retrain) or check RLS/policies for SELECT."
        )
        strict = os.environ.get("GUARDRAIL_SYNC_STRICT", "").strip().lower() in ("1", "true", "yes")
        if strict:
            raise SystemExit(msg)
        print(f"::warning::{msg}", file=sys.stderr)
        raise SystemExit(0)

    payload = build_records_from_row(row)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()

