"""
RemediationDialog (Score Card): enterprise-style dialog with risk, triggers,
suggestions, and REDACT / HASH / ENCRYPT / PROCEED actions.
PyQt6.
"""

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6 import QtCore, QtGui, QtWidgets

from guardrail_runtime import snooze_guard_10_minutes
from infer import highlight_pii_in_text
from pii_remediation import remediate_text

DEEP_SLATE = "#1e1e2e"
TEXT_LIGHT = "#e0e0e0"
FONT_FAMILY = "Segoe UI"


def _risk_level_from_score(score: int) -> str:
    if score < 30:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"


class RemediationDialog(QtWidgets.QDialog):
    """Score card: risk, triggers, suggestions, and remediation buttons."""

    def __init__(
        self,
        original_text: str,
        result: dict,
        llm_target: str,
        parent=None,
    ):
        super().__init__(parent)
        self._original_text = original_text or ""
        self._result = result
        self._llm_target = llm_target or ""
        self.setWindowTitle("PII Risk – Remediation")
        self.setFixedSize(400, 560)
        self.setStyleSheet(
            f"""
            QDialog {{ background: {DEEP_SLATE}; border-radius: 8px; }}
            QLabel {{ color: {TEXT_LIGHT}; font-family: {FONT_FAMILY}; }}
            QPlainTextEdit {{ font-family: Consolas, monospace; font-size: 10px; }}
            QPushButton {{ font-family: {FONT_FAMILY}; padding: 8px 14px; border-radius: 4px; }}
            QPushButton#primary {{ background: #0d6efd; color: white; }}
            QPushButton#secondary {{ background: #6c757d; color: white; }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        score = result.get("risk_score", 0)
        risk = result.get("risk", "low")
        level = _risk_level_from_score(score)

        # Header
        header = QtWidgets.QLabel(f"PII Risk: {level} ({score}/100)")
        header.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {TEXT_LIGHT};")
        layout.addWidget(header)
        target_label = QtWidgets.QLabel(f"Target: {llm_target}")
        target_label.setStyleSheet("font-size: 12px; color: #a0a0a0;")
        layout.addWidget(target_label)

        n_chunks = int(result.get("chunks_scored") or 0)
        if bool(result.get("text_truncated")) and n_chunks > 1:
            chunk_info = QtWidgets.QLabel(
                f"Long text — scanned in {n_chunks} chunks · worst section flagged"
            )
            chunk_info.setWordWrap(True)
            chunk_info.setStyleSheet(
                f"font-size: 10px; color: #6e6e7e; font-family: {FONT_FAMILY}; margin-top: 2px;"
            )
            layout.addWidget(chunk_info)

        spans = result.get("spans") if isinstance(result.get("spans"), list) else []
        if spans:
            fs_head = QtWidgets.QLabel("Flagged spans")
            fs_head.setStyleSheet("font-size: 12px; font-weight: bold; color: #d8d8e8; margin-top: 4px;")
            layout.addWidget(fs_head)
            span_lines = []
            for s in spans:
                if not isinstance(s, dict):
                    continue
                cls = s.get("class", "?")
                src = s.get("source", "")
                try:
                    st = int(s.get("start", 0))
                    en = int(s.get("end", 0))
                except (TypeError, ValueError):
                    st, en = 0, 0
                span_lines.append(f"• {cls} · chars {st}–{en} · {src}")
            sp_lbl = QtWidgets.QLabel("\n".join(span_lines) if span_lines else "—")
            sp_lbl.setWordWrap(True)
            sp_lbl.setStyleSheet(
                "font-size: 10px; color: #8a8a9a; margin-left: 8px; font-family: Consolas, 'Segoe UI', monospace;"
            )
            layout.addWidget(sp_lbl)

            prev_head = QtWidgets.QLabel("Masked preview")
            prev_head.setStyleSheet("font-size: 11px; color: #b0b0b0; margin-top: 6px;")
            layout.addWidget(prev_head)
            preview = QtWidgets.QPlainTextEdit(highlight_pii_in_text(self._original_text, spans))
            preview.setReadOnly(True)
            preview.setMaximumHeight(130)
            preview.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
            preview.setStyleSheet(
                "background: #141414; color: #e0e0e0; border: 1px solid #333; border-radius: 6px; padding: 6px;"
            )
            layout.addWidget(preview)

        # Detected
        layout.addWidget(QtWidgets.QLabel("Detected:"))
        triggers = result.get("triggers", [])
        triggers_text = ", ".join(triggers) if triggers else "None"
        det = QtWidgets.QLabel(triggers_text)
        det.setWordWrap(True)
        det.setStyleSheet("color: #b0b0b0; margin-left: 8px;")
        layout.addWidget(det)

        # Suggestions
        layout.addWidget(QtWidgets.QLabel("Recommendations:"))
        suggestions = result.get("suggestions", [])
        for s in (suggestions or [])[:5]:
            lbl = QtWidgets.QLabel(f"• {s}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("margin-left: 8px; color: #c0c0c0;")
            layout.addWidget(lbl)

        layout.addStretch()

        self._snooze_cb = QtWidgets.QCheckBox("Snooze for 10 minutes")
        self._snooze_cb.setStyleSheet(f"color: {TEXT_LIGHT}; font-family: {FONT_FAMILY};")
        layout.addWidget(self._snooze_cb)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        redact_btn = QtWidgets.QPushButton("REDACT")
        redact_btn.setObjectName("primary")
        redact_btn.clicked.connect(lambda: self._apply("mask"))
        hash_btn = QtWidgets.QPushButton("HASH")
        hash_btn.setObjectName("primary")
        hash_btn.clicked.connect(lambda: self._apply("hash"))
        encrypt_btn = QtWidgets.QPushButton("ENCRYPT")
        encrypt_btn.setObjectName("primary")
        encrypt_btn.clicked.connect(lambda: self._apply("encrypt"))
        proceed_btn = QtWidgets.QPushButton("PROCEED")
        proceed_btn.setObjectName("secondary")
        proceed_btn.clicked.connect(self.accept)

        btn_layout.addWidget(redact_btn)
        btn_layout.addWidget(hash_btn)
        btn_layout.addWidget(encrypt_btn)
        btn_layout.addWidget(proceed_btn)
        layout.addLayout(btn_layout)

    def accept(self) -> None:
        if self._snooze_cb.isChecked():
            snooze_guard_10_minutes()
        super().accept()

    def _apply(self, action: str) -> None:
        safe_text = remediate_text(self._original_text, action)
        clipboard = QtGui.QGuiApplication.clipboard()
        clipboard.setText(safe_text)
        QtWidgets.QMessageBox.information(
            self,
            "Clipboard updated",
            "Safe version copied to clipboard.",
        )
        self.accept()
