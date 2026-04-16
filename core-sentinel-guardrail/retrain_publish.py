from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
SUMMARY_PATH = REPORTS_DIR / "retrain_dashboard_summary.json"
TRAIN_EVAL_SUMMARY_PATH = REPORTS_DIR / "train_eval_summary.json"
PR_CURVE_SUMMARY_PATH = REPORTS_DIR / "pr_curve_summary.json"
THRESHOLD_CALIBRATION_PATH = REPORTS_DIR / "threshold_calibration.json"
RISK_POLICY_PATH = ROOT / "config" / "risk_policy.json"
MODEL_DIR = ROOT / "models" / "tinybert_guardrail"
FEEDBACK_EXPORT_PATH = ROOT / "data" / "user_feedback" / "export.jsonl"
DEFAULT_SUPABASE_TABLE = "retrain_runs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _iso_from_timestamp(ts: float | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _model_version_tag(model_dir: Path) -> str | None:
    if not model_dir.exists():
        return None
    candidates = [
        model_dir / "model.safetensors",
        model_dir / "config.json",
        model_dir / "tokenizer.json",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    newest = max(existing, key=lambda p: p.stat().st_mtime)
    stamp = datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_dir.name}:{stamp}"


def _compact_stage_rows(stage_results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in stage_results or []:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "name": str(row.get("name", "")),
                "status": str(row.get("status", "")),
                "returncode": row.get("returncode"),
                "duration_sec": row.get("duration_sec"),
                "command": row.get("command"),
            }
        )
    return out


def build_retrain_dashboard_summary(
    *,
    stage_results: list[dict[str, Any]] | None = None,
    publish_target: str = "supabase_and_static",
    publish_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_eval = _load_json(TRAIN_EVAL_SUMMARY_PATH)
    pr_curve = _load_json(PR_CURVE_SUMMARY_PATH)
    threshold_calibration = _load_json(THRESHOLD_CALIBRATION_PATH)
    risk_policy = _load_json(RISK_POLICY_PATH)
    best = train_eval.get("best") if isinstance(train_eval.get("best"), dict) else {}
    recommended = (
        threshold_calibration.get("recommended")
        if isinstance(threshold_calibration.get("recommended"), dict)
        else {}
    )
    stages = _compact_stage_rows(stage_results)

    model_exists = MODEL_DIR.is_dir()
    model_mtime = MODEL_DIR.stat().st_mtime if model_exists else None
    failed_stage = next((s["name"] for s in stages if s.get("status") != "success"), None)
    pipeline_status = "success" if not failed_stage else "failed"

    summary: dict[str, Any] = {
        "artifact_version": 1,
        "generated_at": _utc_now_iso(),
        "publish_target": publish_target,
        "pipeline": {
            "status": pipeline_status,
            "failed_stage": failed_stage,
            "stages": stages,
            "completed_stages": [s["name"] for s in stages if s.get("status") == "success"],
        },
        "feedback": {
            "rows_used_for_retrain": _count_jsonl_rows(FEEDBACK_EXPORT_PATH),
            "export_path": FEEDBACK_EXPORT_PATH.as_posix(),
        },
        "model": {
            "path": MODEL_DIR.as_posix(),
            "exists": model_exists,
            "version_tag": _model_version_tag(MODEL_DIR),
            "modified_at": _iso_from_timestamp(model_mtime),
        },
        "metrics": {
            "accuracy": best.get("eval_accuracy"),
            "precision": best.get("eval_precision"),
            "recall": best.get("eval_recall"),
            "f1": best.get("eval_f1"),
            "fpr_non_high_as_high": recommended.get("fpr_non_high_as_high"),
            "precision_low": best.get("eval_precision_low"),
            "recall_low": best.get("eval_recall_low"),
            "f1_low": best.get("eval_f1_low"),
            "precision_high": best.get("eval_precision_high"),
            "recall_high": best.get("eval_recall_high"),
            "f1_high": best.get("eval_f1_high"),
            "precision_med": best.get("eval_precision_med"),
            "recall_med": best.get("eval_recall_med"),
            "f1_med": best.get("eval_f1_med"),
        },
        "evaluation": {
            "train_eval_summary": train_eval,
            "pr_curve_summary": pr_curve,
            "threshold_calibration": threshold_calibration,
        },
        "policy": {
            "block_threshold": risk_policy.get("block_threshold"),
            "risk_score_silent_max": risk_policy.get("risk_score_silent_max"),
            "inference_3class_labels": risk_policy.get("inference_3class_labels"),
            "thresholds": risk_policy.get("thresholds"),
            "calibrated_at": risk_policy.get("_calibrated_at"),
            "calibration_source": risk_policy.get("_calibration_source"),
        },
        "publish": publish_status or {
            "attempted": False,
            "success": False,
            "table": os.environ.get("GUARDRAIL_RETRAIN_SUPABASE_TABLE", DEFAULT_SUPABASE_TABLE),
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _build_supabase_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") or {}
    feedback = summary.get("feedback") or {}
    model = summary.get("model") or {}
    policy = summary.get("policy") or {}
    threshold_cal = ((summary.get("evaluation") or {}).get("threshold_calibration") or {})
    recommended = threshold_cal.get("recommended") if isinstance(threshold_cal, dict) else {}

    return {
        "run_at": summary.get("generated_at"),
        "status": (summary.get("pipeline") or {}).get("status"),
        "model_version_tag": model.get("version_tag"),
        "model_path": model.get("path"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "recall_high": metrics.get("recall_high"),
        "recall_med": metrics.get("recall_med"),
        "feedback_rows": feedback.get("rows_used_for_retrain"),
        "block_threshold": policy.get("block_threshold"),
        "warn_threshold": policy.get("risk_score_silent_max"),
        "recommended_prob_threshold_high": recommended.get("prob_threshold_high"),
        "recommended_prob_threshold_med": recommended.get("prob_threshold_med"),
        "fpr_non_high_as_high": recommended.get("fpr_non_high_as_high"),
        "summary_json": summary,
        "source": "core_sentinel_auto_retrain",
    }


def publish_retrain_summary_to_supabase(summary: dict[str, Any]) -> dict[str, Any]:
    url = (
        os.environ.get("GUARDRAIL_SUPABASE_URL")
        or os.environ.get("SUPABASE_URL")
        or ""
    ).strip().rstrip("/")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("GUARDRAIL_SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    table = (os.environ.get("GUARDRAIL_RETRAIN_SUPABASE_TABLE") or DEFAULT_SUPABASE_TABLE).strip()

    if not url or not key:
        return {
            "attempted": False,
            "success": False,
            "table": table,
            "error": "SUPABASE_URL and key not configured",
        }

    row = _build_supabase_row(summary)
    endpoint = f"{url}/rest/v1/{urllib.parse.quote(table)}"
    body = json.dumps(row).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            response_body = resp.read().decode("utf-8", errors="replace")
        return {
            "attempted": True,
            "success": True,
            "table": table,
            "response": response_body[:500],
        }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "attempted": True,
            "success": False,
            "table": table,
            "error": f"HTTP {exc.code}: {detail[:500]}",
        }
    except Exception as exc:
        return {
            "attempted": True,
            "success": False,
            "table": table,
            "error": str(exc),
        }


__all__ = [
    "SUMMARY_PATH",
    "build_retrain_dashboard_summary",
    "publish_retrain_summary_to_supabase",
]
