"""
Full-height right-edge drawer: Sentinel remediation panel (non-modal).
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import threading
from datetime import datetime, timezone
from functools import partial
from html import escape
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6 import QtCore, QtGui, QtWidgets

import user_settings
from guardrail_runtime import is_monitoring_paused, set_monitoring_paused, snooze_guard_minutes
from infer import critical_secret_spans_for_ui, filter_triggers_already_in_spans
from guardrail_logs import log_remediation_event
from infer import _log_scoring_event
from pii_remediation import (
    _resolve_span_bounds,
    mask_pii_spans,
    redact_for_clipboard,
    remediate_text,
    rephrase_text,
)
from feedback_store import count_pending_wrong_feedback, record_feedback
from toast import Toast, show_toast
from font_clamp import MIN_PX_BODY, paint_font_px

RETRAIN_MIN = max(
    1,
    int(os.environ.get("GUARDRAIL_RETRAIN_MIN_CORRECTIONS", "10") or "10"),
)

FONT_FAMILY = "Segoe UI"
PANEL_W = 408
_PANEL_MAX_H = 900  # cap; also limited to ~85% of available screen height


def _remediation_dialog_height_px(screen: QtGui.QScreen | None) -> int:
    """Fit the remediation drawer: at most 900px tall and at most ~85% of the work area."""
    if screen is None:
        return _PANEL_MAX_H
    screen_h = screen.availableGeometry().height()
    return min(_PANEL_MAX_H, int(screen_h * 0.85))


HEADER_H = 52
HEADER_DRAG_HINT_H = 12
HEADER_TOTAL_H = HEADER_DRAG_HINT_H + HEADER_H
SCORE_SECTION_H = 100
FOOTER_H = 52
ANIM_MS = 220
ACCENT_TEAL = "#02C39A"

_RISK_BADGE_ALL_SKIPPED_QSS = (
    "background: #9E9E9E; color: white; border-radius: 10px; padding: 2px 10px; "
    "font-size: 11px; font-weight: 600; min-width: 72px;"
)

# QComboBox in the drawer (settings + any future combos): popup list must stay dark-on-light.
_PANEL_COMBOBOX_QSS = """
QComboBox {
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 12px;
    min-width: 100px;
    color: #02C39A;
    font-weight: 600;
}
QComboBox QAbstractItemView {
    background-color: white;
    color: #1a1a1a;
    selection-background-color: #02C39A;
    selection-color: white;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 4px;
    font-size: 13px;
}
QComboBox QAbstractItemView::item {
    height: 36px;
    padding-left: 12px;
    color: #1a1a1a;
}
QComboBox QAbstractItemView::item:hover {
    background-color: #f0fdf9;
    color: #02C39A;
}
"""


def _score_tier(score: int) -> tuple[QtGui.QColor, str, str, str]:
    """Line color, score-section label (Safe/Medium/High), header pill bg, header pill text."""
    s = max(0, min(100, int(score)))
    if s <= 40:
        return QtGui.QColor("#2E7D32"), "Safe", "#1B5E20", "Safe"
    if s <= 70:
        return QtGui.QColor("#F57F17"), "Medium", "#E65100", "Medium"
    return QtGui.QColor("#C62828"), "High", "#B71C1C", "High Risk"


def _score_wheel_arc_color(score: int) -> QtGui.QColor:
    """Ring arc color from numeric score only (not span count)."""
    s = max(0, min(100, int(score)))
    if s >= 71:
        return QtGui.QColor("#E53935")
    if s >= 41:
        return QtGui.QColor("#FFB300")
    return QtGui.QColor("#43A047")


def _strip_risk_colors(span_or_trigger: dict | str) -> tuple[QtGui.QColor, str]:
    if isinstance(span_or_trigger, dict):
        r = str(span_or_trigger.get("risk", "low")).lower()
        if r in ("high", "h", "critical", "crit"):
            return QtGui.QColor("#E53935"), "high"
        if r in ("med", "medium", "m"):
            return QtGui.QColor("#F9A825"), "med"
        return QtGui.QColor("#F9A825"), "med"


class MaskingModeSlider(QtWidgets.QWidget):
    mode_changed = QtCore.pyqtSignal(str)

    MODES = [
        ("Partial", "partial", "#02C39A"),
        ("Stars", "stars", "#1565C0"),
        ("Full", "full", "#E65100"),
        ("Type only", "type_only", "#7B1FA2"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._index = 0
        self._drag_x = None
        self.setFixedHeight(52)
        self.setMinimumWidth(260)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setMouseTracking(True)

    def current_mode(self) -> str:
        return self.MODES[self._index][1]

    def current_label(self) -> str:
        return self.MODES[self._index][0]

    def current_color(self) -> str:
        return self.MODES[self._index][2]

    def set_mode(self, mode: str) -> None:
        m = str(mode or "").lower()
        for i, (_, key, _) in enumerate(self.MODES):
            if key == m:
                if i != self._index:
                    self._index = i
                    self.update()
                return

    def _index_from_x(self, x: int) -> int:
        w = max(1, self.width())
        n = len(self.MODES)
        section = w / n
        idx = int(x / section)
        return max(0, min(idx, n - 1))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            new_idx = self._index_from_x(int(event.position().x()))
            if new_idx != self._index:
                self._index = new_idx
                self.update()
                self.mode_changed.emit(self.current_mode())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() == QtCore.Qt.MouseButton.LeftButton:
            new_idx = self._index_from_x(int(event.position().x()))
            if new_idx != self._index:
                self._index = new_idx
                self.update()
                self.mode_changed.emit(self.current_mode())
        super().mouseMoveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        n = len(self.MODES)
        inner_w = max(1.0, float(w - 24))

        track_rect = QtCore.QRectF(12, h // 2 - 3, w - 24, 6)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(220, 220, 220, 180))
        painter.drawRoundedRect(track_rect, 3, 3)

        if n > 1:
            fill_w = (self._index / (n - 1)) * inner_w
        else:
            fill_w = inner_w
        if fill_w > 0:
            color = QtGui.QColor(self.current_color())
            fill_rect = QtCore.QRectF(12, h // 2 - 3, fill_w, 6)
            painter.setBrush(color)
            painter.drawRoundedRect(fill_rect, 3, 3)

        for i, (label, _mode, color) in enumerate(self.MODES):
            if n > 1:
                tick_x = 12 + (i / (n - 1)) * inner_w
            else:
                tick_x = w / 2

            if i == self._index:
                dot_color = QtGui.QColor(color)
                painter.setBrush(dot_color)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2.0))
                painter.drawEllipse(QtCore.QPointF(tick_x, h // 2), 8, 8)
            else:
                painter.setBrush(QtGui.QColor(180, 180, 180))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawEllipse(QtCore.QPointF(tick_x, h // 2), 4, 4)

            label_color = QtGui.QColor(color) if i == self._index else QtGui.QColor(150, 150, 150)
            painter.setPen(label_color)
            font = QtGui.QFont()
            font.setPixelSize(10)
            if i == self._index:
                font.setBold(True)
            painter.setFont(font)
            label_rect = QtCore.QRectF(tick_x - 35, min(h // 2 + 10, h - 18), 70, 16)
            painter.drawText(
                label_rect,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                label,
            )

        if n > 1:
            thumb_x = 12 + (self._index / (n - 1)) * inner_w
        else:
            thumb_x = w / 2
        thumb_color = QtGui.QColor(self.current_color())
        ring_color = QtGui.QColor(thumb_color)
        ring_color.setAlpha(40)
        painter.setBrush(ring_color)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(QtCore.QPointF(thumb_x, h // 2), 13, 13)
        painter.setBrush(thumb_color)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2.5))
        painter.drawEllipse(QtCore.QPointF(thumb_x, h // 2), 9, 9)


class _IssueCardWidget(QtWidgets.QFrame):
    """Single remediation issue row with Fix / Skip and resolved states."""

    def __init__(
        self,
        dialog: "RemediationDialog",
        title: str,
        preview: str,
        strip_col: QtGui.QColor,
        span: dict | None = None,
        trigger_text: str | None = None,
    ):
        super().__init__(dialog)
        self._dialog = dialog
        self._span = span
        self._trigger_text = trigger_text
        self._resolved = False
        self._skipped = False

        self.setMinimumHeight(56)
        self.setStyleSheet("QFrame { background: #ffffff; border: none; }")
        self._opacity_effect: QtWidgets.QGraphicsOpacityEffect | None = None

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 8, 0)
        row.setSpacing(0)

        strip = QtWidgets.QFrame()
        strip.setFixedWidth(3)
        strip.setStyleSheet(f"background: {strip_col.name()}; border: none;")
        row.addWidget(strip)

        mid = QtWidgets.QVBoxLayout()
        mid.setContentsMargins(10, 6, 8, 6)
        mid.setSpacing(2)
        self._title_lbl = QtWidgets.QLabel(title)
        self._title_lbl.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #111827; font-family: Segoe UI;"
        )
        mid.addWidget(self._title_lbl)
        pv = (preview or "—").replace("\n", " ")
        if len(pv) > 30:
            pv = pv[:30] + "…"
        self._preview_lbl = QtWidgets.QLabel(pv)
        self._preview_lbl.setStyleSheet("font-size: 10px; color: #6B7280; font-family: Segoe UI;")
        mid.addWidget(self._preview_lbl)
        row.addLayout(mid, 1)

        self._btn_host = QtWidgets.QWidget()
        btn_lay = QtWidgets.QHBoxLayout(self._btn_host)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(4)

        self._fix_btn = QtWidgets.QPushButton("Fix")
        self._fix_btn.setFixedSize(44, 22)
        self._fix_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._fix_btn.setStyleSheet(
            "QPushButton { background: #02C39A; color: white; font-size: 10px; border: none; "
            "border-radius: 4px; } QPushButton:hover { background: #02a882; }"
        )
        self._fix_btn.clicked.connect(self._on_fix_clicked)

        self._skip_btn = QtWidgets.QPushButton("Skip")
        self._skip_btn.setFixedSize(44, 22)
        self._skip_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._skip_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #6B7280; font-size: 10px; "
            "border: 1px solid #E5E7EB; border-radius: 4px; }"
            "QPushButton:hover { background: #F9FAFB; }"
        )
        self._skip_btn.clicked.connect(self._on_skip_clicked)

        self._fixing_lbl = QtWidgets.QLabel("Fixing...")
        self._fixing_lbl.setStyleSheet("font-size: 11px; color: #9ca3af; font-family: Segoe UI;")
        self._fixing_lbl.setMinimumWidth(72)
        self._fixing_lbl.setVisible(False)

        self._status_lbl = QtWidgets.QLabel("")
        self._status_lbl.setVisible(False)

        btn_lay.addWidget(self._fix_btn)
        btn_lay.addWidget(self._skip_btn)
        btn_lay.addWidget(self._fixing_lbl)
        btn_lay.addWidget(self._status_lbl)
        row.addWidget(self._btn_host)

    def _on_fix_clicked(self) -> None:
        self._dialog._on_card_fix(self)

    def _on_skip_clicked(self) -> None:
        self._dialog._on_card_skip(self)

    def is_actionable(self) -> bool:
        return not self._resolved and not self._skipped

    def set_fixing(self) -> None:
        self._fix_btn.setVisible(False)
        self._skip_btn.setVisible(False)
        self._fixing_lbl.setVisible(True)

    def set_fixed(self) -> None:
        self._resolved = True
        self._fix_btn.setVisible(False)
        self._skip_btn.setVisible(False)
        self._fixing_lbl.setVisible(False)
        self._title_lbl.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #6b7280; font-family: Segoe UI;"
            "text-decoration: line-through;"
        )
        self._status_lbl.setText("✓ Fixed")
        self._status_lbl.setStyleSheet(
            "font-size: 10px; color: #02C39A; font-weight: 600; font-family: Segoe UI;"
        )
        self._status_lbl.setVisible(True)
        self._apply_card_fade(0.3)

    def set_skipped(self) -> None:
        self._skipped = True
        self._fix_btn.setVisible(False)
        self._skip_btn.setVisible(False)
        self._fixing_lbl.setVisible(False)
        self._status_lbl.setText("Skipped")
        self._status_lbl.setStyleSheet(
            "font-size: 10px; color: #757575; font-weight: 500; font-family: Segoe UI;"
        )
        self._status_lbl.setVisible(True)
        self._apply_card_fade(0.3)

    def restore_actionable(self) -> None:
        """Undo bulk skip: restore Fix/Skip and full opacity (only for skipped, non-resolved cards)."""
        if self._resolved:
            return
        self._skipped = False
        self._fix_btn.setVisible(True)
        self._skip_btn.setVisible(True)
        self._fixing_lbl.setVisible(False)
        self._status_lbl.setVisible(False)
        self._status_lbl.setText("")
        self._title_lbl.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #111827; font-family: Segoe UI;"
        )
        self._opacity_effect = None
        self.setGraphicsEffect(None)

    def _apply_card_fade(self, opacity: float) -> None:
        if self._opacity_effect is None:
            self._opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(opacity)


class _HeaderShield(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        fill = QtGui.QColor(ACCENT_TEAL)
        cx, cy = 9.0, 9.0
        sp = QtGui.QPainterPath()
        sp.moveTo(cx, cy - 7)
        sp.lineTo(cx + 7, cy - 3)
        sp.lineTo(cx + 7, cy + 6)
        sp.lineTo(cx - 7, cy + 6)
        sp.lineTo(cx - 7, cy - 3)
        sp.closeSubpath()
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.fillPath(sp, fill)


class _ScoreWheel(QtWidgets.QWidget):
    """64×64 circular progress, 6px arc stroke."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(64, 64)
        self._score = 0

    def set_score(self, score: int) -> None:
        self._score = max(0, min(100, int(score)))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        col = _score_wheel_arc_color(self._score)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        cx, cy = 32.0, 32.0
        r_outer = 28.0
        r_text = 18.0
        pen_w = 6.0

        p.setPen(QtGui.QPen(QtGui.QColor(55, 55, 58), pen_w))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawEllipse(QtCore.QRectF(cx - r_outer, cy - r_outer, 2 * r_outer, 2 * r_outer))

        span = int(5760 * (self._score / 100.0))
        if span > 0:
            p.setPen(QtGui.QPen(col, pen_w))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawArc(
                QtCore.QRectF(cx - r_outer, cy - r_outer, 2 * r_outer, 2 * r_outer),
                90 * 16,
                -span,
            )

        p.setPen(col)
        fam = p.font().family() or "Segoe UI"
        p.setFont(paint_font_px(fam, 20, bold=True, floor=MIN_PX_BODY))
        p.drawText(
            QtCore.QRectF(cx - r_text, cy - 12, 2 * r_text, 24),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            str(self._score),
        )


class RemediationDialog(QtWidgets.QDialog):
    """Full-height drawer from the right; non-modal."""

    monitoring_toggled = QtCore.pyqtSignal(bool)
    remediation_finished = QtCore.pyqtSignal(bool)
    _rephrase_finished = QtCore.pyqtSignal(str)

    def __init__(
        self,
        original_text: str,
        result: dict,
        llm_target: str,
        parent=None,
        *,
        hold_mode: bool = False,
        critical_hold: bool = False,
    ):
        super().__init__(parent)
        _app = QtWidgets.QApplication.instance()
        _screen = _app.primaryScreen() if _app else None
        dialog_h = _remediation_dialog_height_px(_screen)
        self.setFixedSize(PANEL_W, dialog_h)

        self._hold_mode = bool(hold_mode)
        self._critical_hold = bool(critical_hold)
        self._warn_review_mode = False
        self._bubble = parent
        self._mask_mode = "partial"
        self._original_text = original_text or ""
        self._result = result if isinstance(result, dict) else {}
        self._llm_target = llm_target or ""
        self._slide_in_done = False
        self._closing_anim = False
        self._pos_anim: QtCore.QPropertyAnimation | None = None
        self._pending_end_pos: QtCore.QPoint | None = None
        self._dragging = False
        self._drag_offset: QtCore.QPoint | None = None

        self._issue_cards: list[_IssueCardWidget] = []
        self._fixed_spans: list[dict] = []
        self._app_filter_installed = False
        self._slide_out_pending_accept = False
        self._fix_all_running = False
        self._fix_all_queue: list[_IssueCardWidget] = []
        self._preview_height_anim: QtCore.QPropertyAnimation | None = None
        self._preview_copy_text: str = ""
        self._settings_expanded = False
        self._skipped_batch_cards: list = []
        self._undo_skip_timer = QtCore.QTimer(self)
        self._undo_skip_timer.setSingleShot(True)
        self._undo_skip_timer.setInterval(10000)
        self._undo_skip_timer.timeout.connect(self._hide_undo_skip_link)

        self.setModal(False)
        self.setWindowTitle("Sentinel")
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, False)

        try:
            rs = int(self._result.get("risk_score", 0))
        except (TypeError, ValueError):
            rs = 0
        self._risk_score = rs

        spans = self._result.get("spans") if isinstance(self._result.get("spans"), list) else []
        triggers = self._result.get("triggers", [])
        if not isinstance(triggers, (list, tuple)):
            triggers = []
        if (
            not spans
            and not triggers
            and bool(self._result.get("critical_secret_detected"))
            and (self._original_text or "").strip()
        ):
            syn = critical_secret_spans_for_ui(self._original_text.strip())
            if syn:
                spans = syn
                self._result["spans"] = syn
        triggers = filter_triggers_already_in_spans(
            [s for s in spans if isinstance(s, dict)], list(triggers)
        )
        issue_count = len(spans) + len(triggers)
        self._no_literal_spans = not bool(spans) and not bool(triggers)
        if issue_count == 0:
            if rs >= 40:
                display_score = rs
                display_color, display_risk, pill_bg, pill_txt = _score_tier(rs)
            else:
                display_score = 0
                display_risk = "Safe"
                display_color = QtGui.QColor("#2E7D32")
                pill_bg = "#1B5E20"
                pill_txt = "Safe"
        else:
            display_score = rs
            display_color, display_risk, pill_bg, pill_txt = _score_tier(rs)

        self._has_issues = bool(spans) or bool(triggers)
        self._spans = spans
        self._triggers = triggers
        self._last_text = self._original_text
        self._last_spans = list(spans) if isinstance(spans, list) else []
        glob_risk = str(self._result.get("risk", "low")).lower()
        self._glob_risk = glob_risk
        self._current_risk = glob_risk
        self._current_score = rs
        self._correction_picker: QtWidgets.QWidget | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._main_layout = root

        self._hold_banner = QtWidgets.QWidget()
        self._safe_banner = QtWidgets.QWidget()
        self._build_hold_banners()
        root.addWidget(self._hold_banner)
        root.addWidget(self._safe_banner)

        # --- Header (drag hint + title row) ---
        self._header = QtWidgets.QWidget()
        self._header.setFixedHeight(HEADER_TOTAL_H)
        self._header.setStyleSheet(
            "background: #1C1C1E; border: none; border-bottom: 1px solid rgba(255,255,255,0.1);"
        )
        root_header = QtWidgets.QVBoxLayout(self._header)
        root_header.setContentsMargins(0, 0, 0, 0)
        root_header.setSpacing(0)

        drag_hint = QtWidgets.QLabel("• • •")
        drag_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        drag_hint.setFixedHeight(HEADER_DRAG_HINT_H)
        drag_hint.setStyleSheet(
            "color: #666; font-size: 10px; border: none; background: transparent;"
        )
        drag_hint.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        root_header.addWidget(drag_hint)

        header_body = QtWidgets.QWidget()
        header_body.setFixedHeight(HEADER_H)
        hl = QtWidgets.QHBoxLayout(header_body)
        hl.setContentsMargins(10, 0, 8, 0)
        hl.setSpacing(6)

        shield = _HeaderShield(header_body)
        shield.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        hl.addWidget(shield)
        title = QtWidgets.QLabel("Sentinel")
        title.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        title.setStyleSheet(
            f'font-size: 13px; font-weight: bold; color: #ffffff; font-family: "{FONT_FAMILY}";'
        )
        hl.addWidget(title)

        hl.addStretch(1)

        self._risk_badge = QtWidgets.QLabel(pill_txt)
        self._risk_badge.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._risk_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._risk_badge.setStyleSheet(
            f"background: {pill_bg}; color: #ffffff; font-size: 11px; font-weight: bold; "
            f"padding: 4px 10px; border-radius: 12px; font-family: {FONT_FAMILY}; min-width: 72px;"
        )
        self._risk_badge_default_text = pill_txt
        self._risk_badge_default_style = self._risk_badge.styleSheet()
        hl.addWidget(self._risk_badge)

        hl.addStretch(1)

        self._power_btn = QtWidgets.QToolButton()
        self._power_btn.setFixedSize(28, 28)
        self._power_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._power_btn.setText("⏻")
        self._power_btn.clicked.connect(self._toggle_monitoring_pause)
        self._refresh_power_style()
        hl.addWidget(self._power_btn)

        self._gear_btn = QtWidgets.QToolButton()
        self._gear_btn.setFixedSize(28, 28)
        self._gear_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._gear_btn.setText("⚙")
        self._gear_btn.setStyleSheet(
            "QToolButton { border: none; font-size: 15px; color: #c0c0c0; background: transparent; }"
            "QToolButton:hover { color: #ffffff; }"
        )
        self._gear_btn.clicked.connect(self._toggle_settings_accordion)
        hl.addWidget(self._gear_btn)

        collapse_btn = QtWidgets.QToolButton()
        collapse_btn.setFixedSize(28, 28)
        collapse_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        collapse_btn.setText("─")
        collapse_btn.setToolTip("Collapse panel")
        collapse_btn.setStyleSheet(
            "QToolButton { border: none; color: #888; font-size: 14px; background: transparent; }"
            "QToolButton:hover { color: #fff; }"
        )
        collapse_btn.clicked.connect(self.slide_out_and_hide)
        hl.addWidget(collapse_btn)

        close_btn = QtWidgets.QToolButton()
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        close_btn.setText("✕")
        close_btn.setStyleSheet(
            "QToolButton { border: none; color: #888; font-size: 14px; background: transparent; }"
            "QToolButton:hover { color: #fff; }"
        )
        close_btn.clicked.connect(self.slide_out_and_hide)
        hl.addWidget(close_btn)

        root_header.addWidget(header_body)

        self._header_widget = self._header
        self._header_widget.setMouseTracking(True)
        self._header_widget.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
        self._header_widget.mousePressEvent = self._header_mouse_press
        self._header_widget.mouseMoveEvent = self._header_mouse_move
        self._header_widget.mouseReleaseEvent = self._header_mouse_release

        root.addWidget(self._header)

        # --- Score section 100px ---
        score_row = QtWidgets.QWidget()
        score_row.setFixedHeight(SCORE_SECTION_H)
        score_row.setStyleSheet("background: #2C2C2E; border: none;")
        srl = QtWidgets.QHBoxLayout(score_row)
        srl.setContentsMargins(12, 8, 12, 8)
        srl.setSpacing(12)

        self._score_wheel = _ScoreWheel(score_row)
        self._score_wheel.set_score(display_score)
        srl.addWidget(self._score_wheel, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(4)
        lbl_rs = QtWidgets.QLabel("Risk score")
        lbl_rs.setStyleSheet("font-size: 10px; color: #9ca3af; font-family: Segoe UI;")
        right_col.addWidget(lbl_rs)
        self._lbl_tier_big = QtWidgets.QLabel(display_risk)
        self._lbl_tier_big.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {display_color.name()}; font-family: {FONT_FAMILY};"
        )
        right_col.addWidget(self._lbl_tier_big)
        n_issues = len(spans) + len(triggers)  # triggers already deduped vs span classes
        if n_issues <= 0 and self._has_issues:
            n_issues = 1
        self._lbl_issue_count = QtWidgets.QLabel("")
        self._issue_count_default_style = "font-size: 11px; color: #9ca3af; font-family: Segoe UI;"
        self._lbl_issue_count.setStyleSheet(self._issue_count_default_style)
        right_col.addWidget(self._lbl_issue_count)
        right_col.addStretch(1)
        srl.addLayout(right_col, 1)

        root.addWidget(score_row)

        # --- Feedback row (below risk score) ---
        feedback_widget = QtWidgets.QWidget()
        feedback_widget.setStyleSheet("background: #2C2C2E; border: none;")
        feedback_layout = QtWidgets.QHBoxLayout(feedback_widget)
        feedback_layout.setContentsMargins(12, 6, 12, 6)
        feedback_layout.setSpacing(8)

        feedback_lbl = QtWidgets.QLabel("Was this correct?")
        feedback_lbl.setStyleSheet(
            "font-size: 11px; color: #888;"
            "background: transparent;"
        )
        feedback_layout.addWidget(feedback_lbl)

        feedback_layout.addStretch(1)

        self._btn_correct = QtWidgets.QPushButton("✓ Correct")
        self._btn_correct.setFixedHeight(28)
        self._btn_correct.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._btn_correct.setStyleSheet(
            """
            QPushButton {
                background: rgba(67,160,71,0.15);
                color: #43A047;
                border: 1px solid rgba(67,160,71,0.4);
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 12px;
            }
            QPushButton:hover {
                background: rgba(67,160,71,0.25);
            }
            QPushButton:pressed {
                background: rgba(67,160,71,0.4);
            }
            """
        )
        self._btn_correct.clicked.connect(self._on_feedback_correct)
        feedback_layout.addWidget(self._btn_correct)

        self._btn_wrong = QtWidgets.QPushButton("✗ Wrong")
        self._btn_wrong.setFixedHeight(28)
        self._btn_wrong.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._btn_wrong.setStyleSheet(
            """
            QPushButton {
                background: rgba(229,57,53,0.15);
                color: #E53935;
                border: 1px solid rgba(229,57,53,0.4);
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 12px;
            }
            QPushButton:hover {
                background: rgba(229,57,53,0.25);
            }
            """
        )
        self._btn_wrong.clicked.connect(self._on_feedback_wrong)
        feedback_layout.addWidget(self._btn_wrong)

        root.addWidget(feedback_widget)

        # --- Settings accordion (inline; built in _build_settings_section) ---
        self._settings_panel = self._build_settings_section()
        self._settings_panel.setMaximumHeight(0)
        self._settings_panel.setVisible(False)
        root.addWidget(self._settings_panel)

        # --- Issues scroll (white) ---
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: #ffffff; }"
            "QScrollBar:vertical { width: 8px; background: #f0f0f0; }"
            "QScrollBar::handle:vertical { background: #c0c0c0; border-radius: 4px; min-height: 24px; }"
        )

        content_host = QtWidgets.QWidget()
        content_host.setStyleSheet("background: #ffffff;")
        cv = QtWidgets.QVBoxLayout(content_host)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        n_chunks = int(self._result.get("chunks_scored") or 0)
        if bool(self._result.get("text_truncated")) and n_chunks > 1:
            chunk_info = QtWidgets.QLabel(
                f"Long text — {n_chunks} chunks · worst section flagged"
            )
            chunk_info.setWordWrap(True)
            chunk_info.setStyleSheet(
                "font-size: 10px; color: #888; padding: 8px 12px; background: #fafafa;"
            )
            cv.addWidget(chunk_info)

        self._issues_scroll_area: QtWidgets.QScrollArea | None = None
        self._empty_state_wrap: QtWidgets.QWidget | None = None

        if not self._has_issues:
            if self._no_literal_spans and rs >= 70:
                self._show_contextual_warning(cv, rs, glob_risk)
            elif self._no_literal_spans and rs >= 40:
                self._show_contextual_caution(cv, rs)
            else:
                empty_wrap = QtWidgets.QWidget()
                self._empty_state_wrap = empty_wrap
                ev = QtWidgets.QVBoxLayout(empty_wrap)
                ev.setContentsMargins(16, 32, 16, 32)
                ev.setSpacing(12)
                ev.addStretch(1)
                sh = QtWidgets.QLabel("🛡")
                sh.setStyleSheet("font-size: 28px; color: #ccc;")
                sh.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                ev.addWidget(sh)
                t_empty = QtWidgets.QLabel("No PII detected")
                t_empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                t_empty.setStyleSheet("font-size: 13px; color: #9ca3af; font-family: Segoe UI;")
                ev.addWidget(t_empty)
                ev.addStretch(2)
                cv.addWidget(empty_wrap)
        else:
            mask_bar = QtWidgets.QWidget()
            mlay = QtWidgets.QVBoxLayout(mask_bar)
            mlay.setContentsMargins(12, 6, 12, 4)
            mlay.setSpacing(6)
            mode_label = QtWidgets.QLabel("Masking mode")
            mode_label.setStyleSheet(
                "font-size: 11px; font-weight: 600; "
                "color: #666; margin-bottom: 4px;"
            )
            mlay.addWidget(mode_label)
            self._mask_slider = MaskingModeSlider(mask_bar)
            self._mask_slider.setFixedHeight(52)
            self._mask_slider.set_mode(self._mask_mode)
            self._mask_slider.mode_changed.connect(self._on_mask_mode_changed)
            mlay.addWidget(self._mask_slider)
            cv.addWidget(mask_bar)

            first = True
            for s in spans:
                if not isinstance(s, dict):
                    continue
                if os.environ.get("GUARDRAIL_DEBUG_REMEDIATION", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    print(f"[card] building card for span: {s}", flush=True)
                cls = str(s.get("class", "PII"))
                try:
                    st = int(s.get("start", 0))
                    en = int(s.get("end", 0))
                except (TypeError, ValueError):
                    st, en = 0, 0
                resolved = _resolve_span_bounds(self._original_text, s)
                if resolved:
                    _, _, preview = resolved
                else:
                    preview = self._original_text[max(0, st) : max(st, en)]
                span_risk = dict(s)
                span_risk.setdefault("risk", glob_risk)
                strip_col, _ = _strip_risk_colors(span_risk)
                span_copy = dict(s)
                w = _IssueCardWidget(self, cls, preview, strip_col, span=span_copy, trigger_text=None)
                self._issue_cards.append(w)
                if not first:
                    line = QtWidgets.QFrame()
                    line.setFixedHeight(1)
                    line.setStyleSheet("background: #F0F0F0; border: none; max-height: 1px;")
                    cv.addWidget(line)
                first = False
                cv.addWidget(w)

            for t in triggers:
                if not t:
                    continue
                ts = str(t)
                strip_col, _ = _strip_risk_colors({"risk": glob_risk})
                title = ts[:24] + ("…" if len(ts) > 24 else "")
                w = _IssueCardWidget(self, title, ts, strip_col, span=None, trigger_text=ts)
                self._issue_cards.append(w)
                if not first:
                    line = QtWidgets.QFrame()
                    line.setFixedHeight(1)
                    line.setStyleSheet("background: #F0F0F0; border: none;")
                    cv.addWidget(line)
                first = False
                cv.addWidget(w)

        suggestions = self._result.get("suggestions", [])
        for s in (suggestions or [])[:4]:
            lbl = QtWidgets.QLabel(f"• {s}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                "font-size: 10px; color: #6b7280; padding: 4px 12px; font-family: Segoe UI;"
            )
            cv.addWidget(lbl)

        cv.addStretch(1)
        scroll.setWidget(content_host)
        self._issues_scroll_area = scroll
        root.addWidget(scroll, 1)

        # --- Single result preview (Redact / Rephrase / Fix all); between scroll and footer ---
        self._preview_box = QtWidgets.QWidget()
        self._preview_box.setObjectName("previewBox")
        self._preview_box.setMaximumHeight(0)
        self._preview_box.setMinimumHeight(0)
        self._preview_box.hide()
        self._preview_box.setStyleSheet(
            """
            QWidget#previewBox {
                background: #F8F9FA;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
            }
            """
        )
        pv_outer = QtWidgets.QVBoxLayout(self._preview_box)
        pv_outer.setContentsMargins(12, 10, 12, 10)
        pv_outer.setSpacing(6)

        title_row = QtWidgets.QHBoxLayout()
        title_row.setSpacing(8)
        self._preview_title = QtWidgets.QLabel()
        self._preview_title.setStyleSheet(
            "font-size: 11px; color: #999999; font-weight: 600; font-family: Segoe UI;"
        )
        title_row.addWidget(self._preview_title, 1)
        self._preview_close_btn = QtWidgets.QToolButton()
        self._preview_close_btn.setText("✕")
        self._preview_close_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._preview_close_btn.setStyleSheet(
            "QToolButton { border: none; color: #888; font-size: 14px; background: transparent; "
            "min-width: 22px; min-height: 22px; }"
            "QToolButton:hover { color: #333; }"
        )
        self._preview_close_btn.setToolTip("Dismiss")
        self._preview_close_btn.clicked.connect(self._hide_preview_animated)
        title_row.addWidget(self._preview_close_btn, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        pv_outer.addLayout(title_row)

        self._preview_content = QtWidgets.QLabel()
        self._preview_content.setWordWrap(True)
        self._preview_content.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._preview_content.setStyleSheet(
            "font-size: 12px; color: #1a1a1a; font-family: Segoe UI;"
        )
        pv_outer.addWidget(self._preview_content)

        copy_row = QtWidgets.QHBoxLayout()
        copy_row.addStretch(1)
        self._copy_preview_btn = QtWidgets.QPushButton("Copy")
        self._copy_preview_btn.setFixedSize(60, 28)
        self._copy_preview_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._copy_preview_btn.setStyleSheet(
            """
            QPushButton {
                background: #02C39A;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover { background: #019B7A; }
            """
        )
        self._copy_preview_btn.clicked.connect(self._on_copy_preview_clicked)
        copy_row.addWidget(self._copy_preview_btn)
        pv_outer.addLayout(copy_row)

        preview_wrap = QtWidgets.QWidget()
        preview_wrap_l = QtWidgets.QVBoxLayout(preview_wrap)
        preview_wrap_l.setContentsMargins(10, 0, 10, 6)
        preview_wrap_l.setSpacing(0)
        preview_wrap_l.addWidget(self._preview_box)

        root.addWidget(preview_wrap)

        # --- Footer: Redact | Rephrase | Snooze | Fix all + undo link ---
        footer = QtWidgets.QWidget()
        self._footer_widget = footer
        footer.setStyleSheet("background: #FFFFFF; border-top: 1px solid #F0F0F0;")
        fl = QtWidgets.QVBoxLayout(footer)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(0)

        footer_row = QtWidgets.QWidget()
        footer_row.setFixedHeight(FOOTER_H)
        footer_row.setStyleSheet("background: #FFFFFF;")
        fh = QtWidgets.QHBoxLayout(footer_row)
        fh.setContentsMargins(8, 8, 8, 8)
        fh.setSpacing(8)

        self._redact_btn = QtWidgets.QPushButton("Redact")
        self._redact_btn.setFixedSize(76, 36)
        self._redact_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._redact_btn.setStyleSheet(
            "QPushButton { background: #F57F17; color: white; font-size: 12px; font-weight: bold; "
            "border: none; border-radius: 8px; }"
            "QPushButton:hover { background: #EF6C00; }"
        )
        self._redact_btn.clicked.connect(self._on_redact_all)
        fh.addWidget(self._redact_btn)

        self._rephrase_btn = QtWidgets.QPushButton("Rephrase")
        self._rephrase_btn.setFixedSize(84, 36)
        self._rephrase_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._rephrase_btn.setStyleSheet(
            "QPushButton { background: #1565C0; color: white; font-size: 12px; font-weight: bold; "
            "border: none; border-radius: 8px; }"
            "QPushButton:hover { background: #0D47A1; }"
        )
        self._rephrase_btn.clicked.connect(self._on_rephrase)
        fh.addWidget(self._rephrase_btn)

        snooze_wrap = QtWidgets.QWidget()
        snooze_wrap.setFixedSize(96 + 28, 36)
        snooze_lay = QtWidgets.QHBoxLayout(snooze_wrap)
        snooze_lay.setContentsMargins(0, 0, 0, 0)
        snooze_lay.setSpacing(0)
        self._snooze_main_btn = QtWidgets.QPushButton("Snooze 10m")
        self._snooze_main_btn.setFixedSize(96, 36)
        self._snooze_main_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._snooze_main_btn.setStyleSheet(
            "QPushButton { background: #ffffff; color: #02C39A; font-size: 12px; "
            "border: 1.5px solid #02C39A; border-right: none; "
            "border-top-left-radius: 8px; border-bottom-left-radius: 8px; }"
            "QPushButton:hover { background: #E8FDF8; }"
        )
        self._snooze_main_btn.clicked.connect(lambda: self._do_snooze(10.0, "10 minutes"))
        self._snooze_drop_btn = QtWidgets.QToolButton()
        self._snooze_drop_btn.setText("▾")
        self._snooze_drop_btn.setFixedSize(28, 36)
        self._snooze_drop_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._snooze_drop_btn.setStyleSheet(
            "QToolButton { background: #ffffff; color: #02C39A; font-size: 12px; font-weight: bold; "
            "border: 1.5px solid #02C39A; border-top-right-radius: 8px; border-bottom-right-radius: 8px; }"
            "QToolButton:hover { background: #E8FDF8; }"
        )
        self._snooze_drop_btn.setToolTip("Snooze duration")
        self._snooze_drop_btn.clicked.connect(self._show_snooze_menu)
        snooze_lay.addWidget(self._snooze_main_btn, 0)
        snooze_lay.addWidget(self._snooze_drop_btn, 0)
        fh.addWidget(snooze_wrap)

        self._fix_all_btn = QtWidgets.QPushButton("Fix all")
        self._fix_all_btn.setFixedSize(76, 36)
        self._fix_all_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._fix_all_btn.setStyleSheet(
            "QPushButton { background: #02C39A; color: white; font-size: 12px; font-weight: bold; "
            "border: none; border-radius: 8px; }"
            "QPushButton:hover { background: #019B7A; }"
        )
        self._fix_all_btn.clicked.connect(self._on_fix_all_clicked)
        self._fix_all_btn.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._fix_all_btn.customContextMenuRequested.connect(self._on_fix_all_context_menu)
        fh.addWidget(self._fix_all_btn)

        self._skip_all_btn = QtWidgets.QPushButton("Skip all")
        self._skip_all_btn.setFixedSize(76, 36)
        self._skip_all_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._skip_all_btn.setStyleSheet(
            "QPushButton { background: #ffffff; color: #757575; font-size: 12px; font-weight: 600; "
            "border: 1px solid #d1d5db; border-radius: 8px; }"
            "QPushButton:hover { background: #f5f5f5; }"
        )
        self._skip_all_btn.clicked.connect(self._on_skip_all_clicked)
        self._skip_all_btn.setVisible(False)
        fh.addWidget(self._skip_all_btn)

        fl.addWidget(footer_row)

        self._override_confirm = QtWidgets.QWidget()
        self._override_confirm.setVisible(False)
        ocl = QtWidgets.QHBoxLayout(self._override_confirm)
        ocl.setContentsMargins(12, 4, 12, 4)
        self._override_hint = QtWidgets.QLabel(
            "Are you sure? This contains HIGH risk data."
        )
        self._override_hint.setStyleSheet(
            "font-size: 11px; color: #555; font-family: Segoe UI;"
        )
        self._override_hint.setWordWrap(True)
        ocl.addWidget(self._override_hint, 1)
        self._override_confirm_btn = QtWidgets.QPushButton("Confirm")
        self._override_confirm_btn.setFixedSize(72, 28)
        self._override_confirm_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._override_confirm_btn.setStyleSheet(
            "QPushButton { background: #E53935; color: white; font-size: 11px; "
            "border: none; border-radius: 6px; font-weight: 600; }"
            "QPushButton:hover { background: #C62828; }"
        )
        self._override_confirm_btn.clicked.connect(self._on_override_confirmed)
        self._override_cancel_btn = QtWidgets.QPushButton("Cancel")
        self._override_cancel_btn.setFixedSize(72, 28)
        self._override_cancel_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._override_cancel_btn.setStyleSheet(
            "QPushButton { background: #f3f4f6; color: #374151; font-size: 11px; "
            "border: 1px solid #E5E7EB; border-radius: 6px; }"
        )
        self._override_cancel_btn.clicked.connect(self._on_override_cancelled)
        ocl.addWidget(self._override_confirm_btn)
        ocl.addWidget(self._override_cancel_btn)
        fl.addWidget(self._override_confirm)

        self._proceed_row = QtWidgets.QWidget()
        pr = QtWidgets.QHBoxLayout(self._proceed_row)
        pr.setContentsMargins(8, 2, 12, 6)
        pr.addStretch(1)
        self._proceed_anyway_btn = QtWidgets.QPushButton("Proceed anyway")
        self._proceed_anyway_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._proceed_anyway_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #9ca3af; font-size: 11px; "
            "border: none; font-weight: 500; }"
            "QPushButton:hover { color: #6b7280; }"
        )
        self._proceed_anyway_btn.clicked.connect(self._on_proceed_anyway_clicked)
        pr.addWidget(self._proceed_anyway_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self._proceed_row.setVisible(self._hold_mode and not self._critical_hold)
        fl.addWidget(self._proceed_row)

        self._undo_skip_link = QtWidgets.QLabel(
            '<a href="#" style="color:#888;text-decoration:none;">Undo — restore all skipped items</a>'
        )
        self._undo_skip_link.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._undo_skip_link.setOpenExternalLinks(False)
        self._undo_skip_link.setVisible(False)
        self._undo_skip_link.setStyleSheet(
            "font-size: 11px; color: #888; padding: 8px 12px 10px 12px;"
        )
        self._undo_skip_link.linkActivated.connect(lambda _u: self._on_undo_skip_all())
        fl.addWidget(self._undo_skip_link)

        root.addWidget(footer)

        self._rephrase_finished.connect(
            self._on_rephrase_done,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )

        self._setup_shortcuts()

        self._apply_panel_height()
        self._update_issue_count()
        # When there are no literal spans/triggers but score >= 40, keep wheel/header aligned
        # with model risk (contextual path). Only zero the wheel when truly low score.
        if self._has_issues:
            self._update_score_display(self._risk_score, self._glob_risk)
        elif rs >= 40:
            self._update_score_display(rs, self._glob_risk)
        else:
            self._update_score_display(0, "low")

        if self._hold_mode and self._critical_hold:
            for c in self._issue_cards:
                c._skip_btn.setEnabled(False)
                c._skip_btn.setToolTip("Cannot skip critical data")
                c._skip_btn.setStyleSheet(
                    "QPushButton { background: #f3f4f6; color: #9ca3af; font-size: 10px; "
                    "border: 1px solid #E5E7EB; border-radius: 4px; }"
                )

    def _analyze_contextual_triggers(self, text: str, score: int) -> list[dict]:
        """
        When spans=[] but score is high, analyze text for heuristic triggers that may
        explain why the model elevated risk. Each item:
        {trigger, reason, severity, suggestion}
        """
        del score  # reserved for future score-conditioned rules
        triggers: list[dict] = []
        if not text or not str(text).strip():
            return triggers

        name_pattern = re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)
        if name_pattern:
            triggers.append(
                {
                    "trigger": f'Full name: "{name_pattern[0]}"',
                    "reason": (
                        "Full names are personal identifiers that can be used "
                        "to identify individuals"
                    ),
                    "severity": "medium",
                    "suggestion": (
                        'Replace with "the user" or a placeholder like [NAME]'
                    ),
                }
            )

        date_pattern = re.findall(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text
        )
        if date_pattern:
            triggers.append(
                {
                    "trigger": f'Date: "{date_pattern[0]}"',
                    "reason": (
                        "Dates of birth or appointment dates can be identifying "
                        "when combined with other information"
                    ),
                    "severity": "medium",
                    "suggestion": (
                        "Remove specific dates unless essential to the question"
                    ),
                }
            )

        age_pattern = re.findall(
            r"\b(?:age[d]?\s+\d{1,3}|\d{1,3}\s+years?\s+old)\b",
            text,
            re.IGNORECASE,
        )
        if age_pattern:
            triggers.append(
                {
                    "trigger": f'Age phrase: "{age_pattern[0]}"',
                    "reason": (
                        "Age combined with other details can support re-identification"
                    ),
                    "severity": "medium",
                    "suggestion": (
                        "Generalize (e.g. “adult” / “minor”) if age is not required"
                    ),
                }
            )

        return triggers

    def _show_contextual_warning(
        self, layout: QtWidgets.QVBoxLayout, score: int, risk: str
    ) -> None:
        del risk  # reserved for future copy tuning
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)
        outer.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        icon = QtWidgets.QLabel("⚠")
        icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(
            "font-size: 32px; color: #FFB300;"
            "background: transparent;"
        )
        outer.addWidget(icon)

        title = QtWidgets.QLabel("Contextual risk detected")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 14px; font-weight: 600;"
            "color: #FFB300; background: transparent;"
        )
        outer.addWidget(title)

        msg = QtWidgets.QLabel(
            f"The model flagged this text with a risk score "
            f"of {score}/100 based on context and patterns, "
            f"but no specific PII pattern was isolated.\n\n"
            f"This often means the text contains:\n"
            f"• Personal details phrased in natural language\n"
            f"• Names combined with identifying context\n"
            f"• Sensitive business or financial language\n"
            f"• Partial identifiers that suggest PII"
        )
        msg.setWordWrap(True)
        msg.setStyleSheet(
            "font-size: 12px; color: #666;"
            "background: transparent; line-height: 1.6;"
        )
        outer.addWidget(msg)

        why_items = self._analyze_contextual_triggers(self._original_text or "", score)
        if why_items:
            why_hdr = QtWidgets.QLabel("Why the model may have flagged this")
            why_hdr.setStyleSheet(
                "font-size: 12px; font-weight: 700; color: #424242;"
                "background: transparent; margin-top: 4px;"
            )
            outer.addWidget(why_hdr)
            for item in why_items:
                sev = str(item.get("severity", "medium")).lower()
                sev_color = "#E65100" if sev == "high" else "#F57F17"
                trig = QtWidgets.QLabel()
                trig.setWordWrap(True)
                tr = escape(str(item.get("trigger", "")))
                rs = escape(str(item.get("reason", "")))
                sg = escape(str(item.get("suggestion", "")))
                trig.setText(
                    f'<p style="margin:0 0 10px 0;">'
                    f'<span style="color:{sev_color};font-weight:600;">{tr}</span><br/>'
                    f'<span style="color:#555;">{rs}</span><br/>'
                    f'<span style="color:#1565C0;font-size:11px;"><i>Tip: {sg}</i></span>'
                    f"</p>"
                )
                trig.setTextFormat(QtCore.Qt.TextFormat.RichText)
                trig.setStyleSheet("background: transparent;")
                outer.addWidget(trig)

        rec = QtWidgets.QLabel(
            "Recommendation: Review the text manually "
            "before pasting. If it contains names, "
            "numbers, or personal context — consider "
            "rephrasing it."
        )
        rec.setWordWrap(True)
        rec.setStyleSheet(
            """
            font-size: 11px;
            color: #E65100;
            background: #FFF8E1;
            border: 1px solid #FFE0B2;
            border-radius: 6px;
            padding: 8px 10px;
        """
        )
        outer.addWidget(rec)
        outer.addStretch(1)
        layout.addWidget(w)

    def _show_contextual_caution(self, layout: QtWidgets.QVBoxLayout, score: int) -> None:
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(10)
        outer.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        icon = QtWidgets.QLabel("ℹ")
        icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(
            "font-size: 28px; color: #1565C0;"
            "background: transparent;"
        )
        outer.addWidget(icon)

        title = QtWidgets.QLabel("Low-level patterns noticed")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 13px; font-weight: 600;"
            "color: #1565C0; background: transparent;"
        )
        outer.addWidget(title)

        msg = QtWidgets.QLabel(
            f"Risk score: {score}/100\n\n"
            "The model detected mild signals that "
            "may indicate personal or sensitive context. "
            "No specific PII pattern was found.\n\n"
            "The paste has been allowed through. "
            "You can dismiss this panel."
        )
        msg.setWordWrap(True)
        msg.setStyleSheet(
            "font-size: 12px; color: #666;"
            "background: transparent;"
        )
        outer.addWidget(msg)
        outer.addStretch(1)
        layout.addWidget(w)

    def _update_score_display(self, score: int, risk: str | None = None) -> None:
        """Align score wheel, tier label, and header pill with the shown risk (see __init__ empty-state path)."""
        try:
            rs = int(score)
        except (TypeError, ValueError):
            rs = 0
        rs = max(0, min(100, rs))
        self._risk_score = rs
        if risk is not None:
            self._glob_risk = str(risk).lower()
        display_color, display_risk, pill_bg, pill_txt = _score_tier(rs)
        self._score_wheel.set_score(rs)
        self._lbl_tier_big.setText(display_risk)
        self._lbl_tier_big.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {display_color.name()}; font-family: {FONT_FAMILY};"
        )
        self._risk_badge.setText(pill_txt)
        self._risk_badge.setStyleSheet(
            f"background: {pill_bg}; color: #ffffff; font-size: 11px; font-weight: bold; "
            f"padding: 4px 10px; border-radius: 12px; font-family: {FONT_FAMILY}; min-width: 72px;"
        )
        self._current_score = rs
        self._current_risk = self._glob_risk

    def _on_feedback_correct(self) -> None:
        """User confirmed the detection was correct."""
        self._record_feedback(
            predicted=self._current_risk,
            correct=self._current_risk,
            feedback_type="correct",
        )
        self._btn_correct.setText("✓ Thanks!")
        self._btn_correct.setEnabled(False)
        self._btn_wrong.setEnabled(False)
        self._btn_correct.setStyleSheet(
            """
            QPushButton {
                background: rgba(67,160,71,0.3);
                color: #43A047;
                border: 1px solid #43A047;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 12px;
            }
            """
        )
        show_toast(
            "Feedback recorded — model gets smarter!",
            color="#43A047",
            duration=2000,
            parent=self,
            title="✓ Confirmed correct",
        )

    def _on_feedback_wrong(self) -> None:
        """User says detection was wrong — pick correct label."""
        self._btn_correct.setEnabled(False)
        self._btn_wrong.setEnabled(False)
        self._show_correction_picker()

    def _show_correction_picker(self) -> None:
        """Inline picker for correct risk label."""
        if self._correction_picker is not None:
            self._main_layout.removeWidget(self._correction_picker)
            self._correction_picker.deleteLater()
            self._correction_picker = None

        picker = QtWidgets.QWidget()
        picker.setObjectName("correctionPicker")
        picker.setStyleSheet(
            """
            QWidget#correctionPicker {
                background: #FFF8E1;
                border: 1px solid #FFB300;
                border-radius: 8px;
            }
            """
        )
        layout = QtWidgets.QVBoxLayout(picker)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        lbl = QtWidgets.QLabel("What should the correct level be?")
        lbl.setStyleSheet(
            "font-size: 12px; font-weight: 600;"
            "color: #E65100; background: transparent;"
        )
        layout.addWidget(lbl)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)

        options = [
            ("Safe", "low", "#43A047"),
            ("Medium", "med", "#FFB300"),
            ("High", "high", "#E53935"),
            ("Critical", "critical", "#B71C1C"),
        ]

        for label, value, color in options:
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(30)
            btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background: {color}22;
                    color: {color};
                    border: 1px solid {color}66;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: 600;
                    padding: 0 10px;
                }}
                QPushButton:hover {{
                    background: {color}44;
                }}
                """
            )
            btn.clicked.connect(partial(self._on_correction_selected, value, label))
            btn_row.addWidget(btn)

        layout.addLayout(btn_row)

        idx = self._main_layout.indexOf(self._footer_widget)
        self._main_layout.insertWidget(idx, picker)
        self._correction_picker = picker

    def _on_correction_selected(self, correct_label: str, display_label: str) -> None:
        """User picked the correct classification."""
        predicted = self._current_risk or "high"

        self._record_feedback(
            predicted=predicted,
            correct=correct_label,
            feedback_type="wrong",
        )

        if self._correction_picker is not None:
            self._main_layout.removeWidget(self._correction_picker)
            self._correction_picker.deleteLater()
            self._correction_picker = None

        self._btn_wrong.setText(f"✗ Corrected → {display_label}")
        self._btn_wrong.setStyleSheet(
            """
            QPushButton {
                background: rgba(255,179,0,0.2);
                color: #FFB300;
                border: 1px solid #FFB300;
                border-radius: 6px;
                font-size: 11px;
                padding: 0 12px;
            }
            """
        )
        show_toast(
            f"Correction saved: {predicted} → {display_label}. Model will learn!",
            color="#FFB300",
            duration=3000,
            parent=self,
            title="Feedback recorded",
        )
        self._check_retrain_threshold()

    def _record_feedback(self, *, predicted: str, correct: str, feedback_type: str) -> None:
        """Persist feedback and sync to Supabase in the background."""
        text = getattr(self, "_original_text", "") or ""
        record_feedback(
            text,
            str(predicted),
            str(correct),
            source="remediation_panel",
            risk_score=int(self._current_score or 0),
            feedback_type=feedback_type,
        )
        print(
            f"[feedback] saved: {predicted} → {correct} ({feedback_type})",
            flush=True,
        )
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        entry = {
            "timestamp": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "text_hash": text_hash,
            "predicted": str(predicted),
            "correct": str(correct),
            "feedback_type": feedback_type,
        }
        threading.Thread(
            target=self._sync_feedback_supabase,
            args=(entry,),
            daemon=True,
        ).start()

    def _sync_feedback_supabase(self, entry: dict) -> None:
        """Sync feedback to Supabase feedback_corrections table when configured."""
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        if not url or not key:
            return
        try:
            from supabase import create_client

            sb = create_client(url, key)
            base = {
                "text_hash": entry["text_hash"],
                "predicted": entry["predicted"],
                "correct": entry["correct"],
                "source": entry["feedback_type"],
                "feedback_type": entry["feedback_type"],
                "used_for_training": False,
            }
            payloads = [
                {"timestamp": entry["timestamp"], **base},
                {"recorded_at": entry["timestamp"], **base},
                dict(base),
            ]
            last_err: Exception | None = None
            for payload in payloads:
                try:
                    sb.table("feedback_corrections").insert(payload).execute()
                    return
                except Exception as e:
                    last_err = e
            if last_err is not None:
                err_s = str(last_err)
                print(f"[feedback] supabase sync failed: {last_err}", flush=True)
                if "predicted" in err_s and "PGRST204" in err_s:
                    print(
                        "[feedback] Your Supabase table `feedback_corrections` is missing columns "
                        "the app expects (predicted, correct, …). Run the SQL in "
                        "core-sentinel-guardrail/supabase/feedback_corrections.sql "
                        "in the Supabase SQL Editor, then reload the API schema.",
                        flush=True,
                    )
        except Exception as e:
            print(f"[feedback] supabase client error: {e}", flush=True)

    def _check_retrain_threshold(self) -> None:
        """Notify when enough wrong-label corrections are pending."""
        pending = count_pending_wrong_feedback()
        print(f"[feedback] {pending} pending corrections", flush=True)
        if pending >= RETRAIN_MIN:
            show_toast(
                f"{pending} corrections collected. Run train.py tonight to retrain!",
                color="#1565C0",
                duration=6000,
                parent=self,
                title="Model update ready",
            )
        elif pending >= 10:
            show_toast(
                f"{pending}/{RETRAIN_MIN} corrections collected.",
                color="#888888",
                duration=2000,
                parent=self,
                title="Feedback progress",
            )

    def show_with_hold_mode(
        self,
        *,
        spans: list | None = None,
        score: int = 0,
        risk: str = "low",
        critical: bool = False,
        original_text: str = "",
    ) -> None:
        """
        Hold paste: sync snapshot fields, refresh score UI, then show.

        Issue cards are built in ``__init__`` from ``result`` (must use a deep-copied result from
        the scorer so ``spans`` are not shared with the inference cache). This updates
        ``_result`` / ``_original_text`` if you pass them, then re-runs score display.
        """
        if os.environ.get("GUARDRAIL_DEBUG_REMEDIATION", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            sp = list(spans) if spans is not None else []
            print(
                f"[panel] hold_mode spans={len(sp)} score={score} risk={risk!r} "
                f"critical={critical}",
                flush=True,
            )
            for s in sp[:3]:
                print(f"  span: {s}", flush=True)

        if original_text:
            self._original_text = original_text
        self._hold_mode = True
        self._critical_hold = bool(critical)
        if spans is not None:
            self._spans = list(spans)
            self._result["spans"] = self._spans
            self._last_spans = list(self._spans)
            self._has_issues = bool(self._spans) or bool(self._triggers)
        self._update_score_display(int(score), risk)
        self._update_issue_count()
        self._skip_all_btn.setVisible(not bool(critical))
        self.show()

    def _show_banner(self, text: str, text_color: str, bg_color: str) -> None:
        """Top strip banner with configurable text and colors (warn / review)."""
        self._hide_hold_banner()
        self._safe_banner.setFixedHeight(0)
        self._safe_banner.setVisible(False)
        if not hasattr(self, "_banner_widget"):
            self._banner_widget = QtWidgets.QWidget()
            self._banner_widget.setFixedHeight(36)
            banner_layout = QtWidgets.QHBoxLayout(self._banner_widget)
            banner_layout.setContentsMargins(12, 0, 12, 0)
            self._banner_icon = QtWidgets.QLabel()
            self._banner_icon.setStyleSheet("font-size: 14px; background: transparent;")
            self._banner_text = QtWidgets.QLabel()
            self._banner_text.setWordWrap(True)
            self._banner_text.setStyleSheet(
                f"font-size: 12px; font-weight: 600; background: transparent; "
                f"font-family: {FONT_FAMILY};"
            )
            banner_layout.addWidget(self._banner_icon)
            banner_layout.addWidget(self._banner_text, 1)
            banner_layout.addStretch()
            self._main_layout.insertWidget(0, self._banner_widget)
        self._banner_widget.setStyleSheet(
            f"background: {bg_color}; border: none; border-radius: 0px;"
        )
        self._banner_icon.setText("⊘" if "blocked" in text.lower() else "⚠")
        self._banner_icon.setStyleSheet(
            f"color: {text_color}; font-size: 14px; background: transparent;"
        )
        self._banner_text.setText(text)
        self._banner_text.setStyleSheet(
            f"color: {text_color}; font-size: 12px; font-weight: 600; "
            f"background: transparent; font-family: {FONT_FAMILY};"
        )
        self._banner_widget.setFixedHeight(36)
        self._banner_widget.show()

    def show_with_warn_mode(
        self,
        *,
        spans: list | None = None,
        score: int = 0,
        risk: str = "low",
    ) -> None:
        """Paste already went through; show panel for review (non-hold): amber banner, no proceed row."""
        self._warn_review_mode = True
        self._hold_mode = False
        self._critical_hold = False
        if spans is not None:
            self._spans = list(spans)
            self._result["spans"] = self._spans
            self._has_issues = bool(self._spans) or bool(self._triggers)
        try:
            sc = int(score)
        except (TypeError, ValueError):
            sc = 0
        rk = str(risk or "low").lower()
        self._update_score_display(sc, rk)
        self._update_issue_count()
        self._proceed_row.setVisible(False)
        self._proceed_anyway_btn.setVisible(False)
        if rk == "high" or sc >= 70:
            self._show_banner(
                "Review before sending — PII detected",
                "#E65100",
                "#FFF8E1",
            )
        else:
            self._show_banner(
                "Heads up — sensitive content detected",
                "#F9A825",
                "#FFFDE7",
            )
        if not hasattr(self, "_fix_all_btn_label_default"):
            self._fix_all_btn_label_default = self._fix_all_btn.text()
            self._fix_all_btn_tip_default = self._fix_all_btn.toolTip()
        self._fix_all_btn.setText("Redact & Copy")
        self._fix_all_btn.setToolTip(
            "Redact PII and copy clean version to clipboard for manual paste"
        )
        self._skip_all_btn.setVisible(True)
        self.show()

    def _build_hold_banners(self) -> None:
        if hasattr(self, "_banner_widget"):
            self._banner_widget.hide()
            self._banner_widget.setFixedHeight(0)
        self._hold_banner.setFixedHeight(0)
        self._hold_banner.setVisible(False)
        self._safe_banner.setFixedHeight(0)
        self._safe_banner.setVisible(False)
        if not self._hold_mode:
            return
        critical = self._critical_hold
        banner_bg = "#FFEBEE" if critical else "#FFF8E1"
        banner_color = "#B71C1C" if critical else "#E65100"
        banner_text = (
            "Paste blocked — fix required"
            if critical
            else "High risk — fix before pasting"
        )
        self._hold_banner.setFixedHeight(36)
        self._hold_banner.setVisible(True)
        self._hold_banner.setStyleSheet(f"background: {banner_bg}; border: none; border-radius: 0px;")
        hbl = QtWidgets.QHBoxLayout(self._hold_banner)
        hbl.setContentsMargins(12, 0, 12, 0)
        icon = QtWidgets.QLabel("⊘" if critical else "⚠")
        icon.setStyleSheet(f"color:{banner_color};font-size:14px;")
        text = QtWidgets.QLabel(banner_text)
        text.setStyleSheet(
            f"color:{banner_color};font-size:12px;font-weight:600;font-family: {FONT_FAMILY};"
        )
        hbl.addWidget(icon)
        hbl.addWidget(text)
        hbl.addStretch(1)

        self._safe_banner.setStyleSheet("background: #E8F5E9; border: none; border-radius: 0px;")
        sbl = QtWidgets.QHBoxLayout(self._safe_banner)
        sbl.setContentsMargins(12, 0, 12, 0)
        self._safe_banner_lbl = QtWidgets.QLabel("")
        self._safe_banner_lbl.setStyleSheet(
            f"color:#2E7D32;font-size:12px;font-weight:600;font-family: {FONT_FAMILY};"
        )
        sbl.addWidget(self._safe_banner_lbl)

    def _copy_to_clipboard_str(self, text: str) -> None:
        try:
            import pyperclip

            pyperclip.copy(text or "")
        except Exception:
            QtGui.QGuiApplication.clipboard().setText(text or "")

    def _auto_paste(self) -> None:
        try:
            if sys.platform == "win32":
                from win_paste_hook import replay_suppressed_paste

                replay_suppressed_paste()
                return
        except Exception:
            pass
        try:
            import pyautogui

            pyautogui.hotkey("ctrl", "v")
        except ImportError:
            show_toast(
                "Safe version in clipboard — paste now",
                color="#02C39A",
                parent=self,
            )

    def _hide_hold_banner(self) -> None:
        if hasattr(self, "_banner_widget"):
            self._banner_widget.hide()
            self._banner_widget.setFixedHeight(0)
        self._hold_banner.setFixedHeight(0)
        self._hold_banner.setVisible(False)

    def _show_safe_banner(self, message: str) -> None:
        self._safe_banner_lbl.setText(message)
        self._safe_banner.setFixedHeight(36)
        self._safe_banner.setVisible(True)

    def _hold_safe_text_from_fixed(self) -> str:
        if self._fixed_spans:
            return mask_pii_spans(
                self._original_text,
                self._fixed_spans,
                mode=self._current_mask_mode(),
            )
        return remediate_text(self._original_text, "mask")

    def _hold_all_issues_fixed(self) -> bool:
        if not self._issue_cards:
            return False
        return all(c._resolved for c in self._issue_cards)

    def _hold_complete_safe_flow(self, safe_text: str, toast_msg: str) -> None:
        self._copy_to_clipboard_str(safe_text)
        show_toast(toast_msg, color="#02C39A", parent=self)
        self._hide_hold_banner()
        self._show_safe_banner("Safe to paste now ✓")
        b = self._bubble
        if b is not None and hasattr(b, "set_hold_state"):
            b.set_hold_state(False)
        QtCore.QTimer.singleShot(800, self._auto_paste)

    def _on_proceed_anyway_clicked(self) -> None:
        if not self._hold_mode or self._critical_hold:
            return
        self._override_confirm.setVisible(True)
        self._proceed_anyway_btn.setVisible(False)

    def _on_override_cancelled(self) -> None:
        self._override_confirm.setVisible(False)
        self._proceed_anyway_btn.setVisible(True)

    def _on_override_confirmed(self) -> None:
        try:
            log_remediation_event(
                "user_override",
                {
                    "action": "user_override",
                    "risk": str(self._result.get("risk", "unknown")),
                    "risk_score": self._result.get("risk_score"),
                },
                text=self._original_text or "",
            )
        except Exception:
            pass
        self._copy_to_clipboard_str(self._original_text)
        show_toast("Override — original content pasted", color="#E53935", parent=self)
        if self._bubble is not None:
            self._bubble.set_hold_state(False)
        self._slide_out_pending_accept = False
        QtCore.QTimer.singleShot(120, self._auto_paste)
        self.slide_out_and_hide()

    def _apply_panel_height(self) -> None:
        screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            app = QtWidgets.QApplication.instance()
            screen = app.primaryScreen() if app else None
        h = _remediation_dialog_height_px(screen)
        self.setFixedSize(PANEL_W, h)

    def _build_settings_section(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(0)
        widget.setStyleSheet("background: #F5F5F6; border-bottom: 1px solid #E0E0E0;")

        settings_data = [
            ("monitor_llm_only", "Monitor LLM windows only", True),
            ("block_high_risk_pastes", "Block high-risk pastes", True),
            ("show_badge_on_issues", "Show badge count", True),
            ("auto_redact_on_block", "Auto-redact on block", False),
            ("play_sound_on_block", "Sound on block", False),
        ]

        saved = self._load_settings()

        for key, label, default in settings_data:
            row = QtWidgets.QWidget()
            row.setFixedHeight(48)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet("color: #1a1a1a; font-size: 13px;")

            toggle = QtWidgets.QCheckBox()
            toggle.setChecked(bool(saved.get(key, default)))
            toggle.setStyleSheet(
                """
                QCheckBox::indicator {
                    width: 44px;
                    height: 24px;
                    border-radius: 12px;
                    border: 2px solid #ccc;
                    background: #ccc;
                }
                QCheckBox::indicator:checked {
                    background: #02C39A;
                    border-color: #02C39A;
                }
                """
            )
            toggle.stateChanged.connect(
                lambda state, k=key: self._save_setting(k, bool(state))
            )

            row_layout.addWidget(lbl)
            row_layout.addStretch()
            row_layout.addWidget(toggle)

            divider = QtWidgets.QFrame()
            divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            divider.setStyleSheet("color: #f0f0f0;")

            layout.addWidget(row)
            layout.addWidget(divider)

        sens_row = QtWidgets.QWidget()
        sens_row.setFixedHeight(48)
        sens_layout = QtWidgets.QHBoxLayout(sens_row)
        sens_layout.setContentsMargins(0, 0, 0, 0)

        sens_lbl = QtWidgets.QLabel("Sensitivity")
        sens_lbl.setStyleSheet("color: #1a1a1a; font-size: 13px;")

        sens_combo = QtWidgets.QComboBox()
        sens_combo.addItems(["Low", "Medium", "High", "Strict"])
        sens_key = str(saved.get("sensitivity", "medium")).lower()
        sens_labels = {"low": "Low", "medium": "Medium", "high": "High", "strict": "Strict"}
        sens_combo.setCurrentText(sens_labels.get(sens_key, "Medium"))
        sens_combo.setStyleSheet(_PANEL_COMBOBOX_QSS)
        sens_combo.currentTextChanged.connect(lambda v: self._save_setting("sensitivity", v))

        sens_layout.addWidget(sens_lbl)
        sens_layout.addStretch()
        sens_layout.addWidget(sens_combo)
        layout.addWidget(sens_row)

        return widget

    def _load_settings(self) -> dict:
        return user_settings.load()

    def _save_setting(self, key: str, value: object) -> None:
        try:
            data = user_settings.load()
            if key == "sensitivity" and isinstance(value, str):
                data["sensitivity"] = value.lower()
            else:
                data[key] = value
            user_settings.save(data)
        except Exception as e:
            print(f"[settings] save failed: {e}")

    def _toggle_settings_accordion(self) -> None:
        self._settings_expanded = not self._settings_expanded
        tgt_h = 400
        dur_ms = 200
        if self._settings_expanded:
            self._settings_panel.setMaximumHeight(0)
            self._settings_panel.setVisible(True)
            anim = QtCore.QPropertyAnimation(self._settings_panel, b"maximumHeight", self)
            anim.setDuration(dur_ms)
            anim.setStartValue(0)
            anim.setEndValue(tgt_h)
            anim.start()
            self._settings_anim = anim
        else:
            start_h = self._settings_panel.maximumHeight()
            if start_h <= 0:
                start_h = tgt_h
            anim = QtCore.QPropertyAnimation(self._settings_panel, b"maximumHeight", self)
            anim.setDuration(dur_ms)
            anim.setStartValue(start_h)
            anim.setEndValue(0)

            def _hide_panel() -> None:
                self._settings_panel.setVisible(False)

            anim.finished.connect(_hide_panel)
            anim.start()
            self._settings_anim = anim

    def _restore_risk_badge_default(self) -> None:
        self._risk_badge.setText(self._risk_badge_default_text)
        self._risk_badge.setStyleSheet(self._risk_badge_default_style)

    def _update_issue_count(self) -> None:
        if self._fix_all_running:
            return
        if not self._issue_cards:
            try:
                rs_ic = int(self._risk_score)
            except (TypeError, ValueError):
                rs_ic = 0
            if getattr(self, "_no_literal_spans", False):
                if rs_ic >= 70:
                    self._lbl_issue_count.setText("Contextual risk")
                    self._lbl_issue_count.setStyleSheet(
                        "font-size: 11px; color: #FFB300; font-weight: 600; font-family: Segoe UI;"
                    )
                    self._restore_risk_badge_default()
                    return
                if rs_ic >= 40:
                    self._lbl_issue_count.setText("Low-level signals")
                    self._lbl_issue_count.setStyleSheet(
                        "font-size: 11px; color: #1565C0; font-weight: 600; font-family: Segoe UI;"
                    )
                    self._restore_risk_badge_default()
                    return
            self._lbl_issue_count.setText("No issues")
            self._lbl_issue_count.setStyleSheet(self._issue_count_default_style)
            self._restore_risk_badge_default()
            return
        n_total = len(self._issue_cards)
        n_fixed = sum(1 for c in self._issue_cards if c._resolved)
        n_skipped = sum(1 for c in self._issue_cards if c._skipped)

        if n_fixed == n_total:
            self._lbl_issue_count.setText("All fixed ✓")
            self._lbl_issue_count.setStyleSheet(
                "font-size: 11px; color: #02C39A; font-weight: bold; font-family: Segoe UI;"
            )
            self._restore_risk_badge_default()
            return
        if n_skipped == n_total:
            self._lbl_issue_count.setText("All skipped")
            self._lbl_issue_count.setStyleSheet(
                "font-size: 11px; color: #757575; font-weight: 600; font-family: Segoe UI;"
            )
            self._risk_badge.setText("All skipped")
            self._risk_badge.setStyleSheet(_RISK_BADGE_ALL_SKIPPED_QSS)
            return
        if n_fixed == 0 and n_skipped == 0:
            self._lbl_issue_count.setText(
                f"{n_total} issue{'s' if n_total != 1 else ''} found"
            )
            self._lbl_issue_count.setStyleSheet(self._issue_count_default_style)
            self._restore_risk_badge_default()
            return
        self._lbl_issue_count.setText(f"{n_fixed} fixed, {n_skipped} skipped")
        self._lbl_issue_count.setStyleSheet(self._issue_count_default_style)
        self._restore_risk_badge_default()

    def _setup_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self, activated=self._on_fix_all_clicked)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Delete"), self, activated=self._on_skip_all_clicked)
        QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+S"),
            self,
            activated=lambda: self._do_snooze(10.0, "10 minutes"),
        )

    def _on_skip_all_clicked(self) -> None:
        if self._fix_all_running:
            return
        if self._hold_mode and self._critical_hold:
            return
        if self._hold_mode and not self._critical_hold:
            show_toast(
                "Skipped — pasting original risky content",
                color="#E53935",
                duration=3000,
            )
            self._copy_to_clipboard_str(self._original_text)
            if self._bubble is not None:
                self._bubble.set_hold_state(False)
            QtCore.QTimer.singleShot(150, self._auto_paste)
            QtCore.QTimer.singleShot(400, self.slide_out_and_hide)
            return
        todo = [c for c in self._issue_cards if c.is_actionable()]
        if not todo:
            return
        self._skipped_batch_cards = list(todo)
        for card in todo:
            card.set_skipped()
        self._update_issue_count()
        total = len(self._issue_cards)
        show_toast(f"Skipped all {total} items", color="#757575", duration=2000)
        self._undo_skip_link.setVisible(True)
        self._undo_skip_link.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self._undo_skip_timer.stop()
        self._undo_skip_timer.start()

    def _on_undo_skip_all(self) -> None:
        for card in self._skipped_batch_cards:
            card.restore_actionable()
        self._skipped_batch_cards = []
        self._undo_skip_link.setVisible(False)
        self._undo_skip_timer.stop()
        self._restore_risk_badge_default()
        self._update_issue_count()

    def _hide_undo_skip_link(self) -> None:
        self._undo_skip_link.setVisible(False)

    def _update_redact_preview(self) -> None:
        if not getattr(self, "_preview_box", None) or not self._preview_box.isVisible():
            return
        if not str(self._preview_title.text()).startswith("Masked version"):
            return
        raw = self._original_text or ""
        spans = self._result.get("spans") if isinstance(self._result.get("spans"), list) else []
        mode = self._current_mask_mode()
        t = mask_pii_spans(raw, spans, mode=mode)
        if t == raw:
            t = remediate_text(raw, "mask")
        self._preview_copy_text = t or ""
        self._preview_content.setText((t or "")[:300])

    def _on_mask_mode_changed(self, mode: str) -> None:
        self._mask_mode = mode
        self._update_redact_preview()
        labels = {
            "partial": "Partial — shows last digits",
            "stars": "Stars — full replacement",
            "full": "Full — type label only",
            "type_only": "Type only — [CLASS]",
        }
        show_toast(
            labels.get(mode, mode),
            color=self._mask_slider.current_color(),
            duration=1500,
            parent=self,
        )

    def _current_mask_mode(self) -> str:
        if hasattr(self, "_mask_slider"):
            return self._mask_slider.current_mode()
        return getattr(self, "_mask_mode", "partial")

    def _clipboard_from_fixed_spans(self) -> None:
        if not self._fixed_spans:
            return
        t = mask_pii_spans(
            self._original_text, self._fixed_spans, mode=self._current_mask_mode()
        )
        QtGui.QGuiApplication.clipboard().setText(t)

    def _clipboard_after_trigger_fix(self) -> None:
        t = remediate_text(self._original_text, "mask")
        QtGui.QGuiApplication.clipboard().setText(t)

    def _on_card_fix(self, card: _IssueCardWidget) -> None:
        if not card.is_actionable() or self._fix_all_running:
            return
        card.set_fixing()
        QtCore.QTimer.singleShot(400, lambda c=card: self._finish_card_fix(c))

    def _finish_card_fix(self, card: _IssueCardWidget) -> None:
        if self._fix_all_running:
            return
        if not card._fixing_lbl.isVisible():
            return
        if card._resolved or card._skipped:
            return
        if card._span is not None:
            self._fixed_spans.append(card._span)
            self._clipboard_from_fixed_spans()
        else:
            self._clipboard_after_trigger_fix()
        card.set_fixed()
        self._update_issue_count()
        if self._hold_mode:
            if self._hold_all_issues_fixed():
                t = self._hold_safe_text_from_fixed()
                self._hold_complete_safe_flow(t, "All fixed — safe to paste")
            return
        show_toast("Fixed — safe version copied", color="#02C39A")

    def _on_card_skip(self, card: _IssueCardWidget) -> None:
        if not card.is_actionable() or self._fix_all_running:
            return
        if self._hold_mode and self._critical_hold:
            return
        if self._hold_mode and not self._critical_hold:
            show_toast(
                "Skipped — pasting original risky content",
                color="#E53935",
                duration=3000,
            )
            self._copy_to_clipboard_str(self._original_text)
            if self._bubble is not None:
                self._bubble.set_hold_state(False)
            QtCore.QTimer.singleShot(150, self._auto_paste)
            QtCore.QTimer.singleShot(400, self.slide_out_and_hide)
            return
        card.set_skipped()
        self._update_issue_count()
        show_toast("Skipped", color="#757575", duration=1500)

    def _set_footer_actions_enabled(self, enabled: bool) -> None:
        self._redact_btn.setEnabled(enabled)
        self._rephrase_btn.setEnabled(enabled)
        self._snooze_main_btn.setEnabled(enabled)
        self._snooze_drop_btn.setEnabled(enabled)
        self._fix_all_btn.setEnabled(enabled)
        self._skip_all_btn.setEnabled(enabled)

    def _on_fix_all_context_menu(self, pos: QtCore.QPoint) -> None:
        if not self._has_issues:
            return
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("Skip all")
        act.triggered.connect(self._on_skip_all_clicked)
        menu.exec(self._fix_all_btn.mapToGlobal(pos))

    def _on_redact_all(self) -> None:
        if self._fix_all_running:
            return
        try:
            log_remediation_event("redact_and_copy", {}, text=self._original_text or "")
        except Exception:
            pass
        raw = self._original_text or ""
        spans = self._result.get("spans") if isinstance(self._result.get("spans"), list) else []
        t = mask_pii_spans(raw, spans, mode="partial")
        if t == raw:
            t = remediate_text(raw, "mask")
        if self._hold_mode and self._has_issues:
            self._hold_complete_safe_flow(t, "Fixed — safe version ready to paste")
            return
        self._show_preview(t, "Masked version (partial):")

    def _on_rephrase(self) -> None:
        if self._fix_all_running:
            return
        text = self._last_text or ""
        spans = self._last_spans if isinstance(self._last_spans, list) else []
        if not str(text).strip():
            show_toast("No text to rephrase", color="#999")
            return
        try:
            log_remediation_event("rephrase_and_copy", {}, text=text)
        except Exception:
            pass
        self._rephrase_btn.setText("...")
        self._rephrase_btn.setEnabled(False)

        def _run() -> None:
            try:
                result = rephrase_text(text, spans)
            except Exception as e:
                print(f"[rephrase] {e}")
                result = ""
            self._rephrase_finished.emit(result or "")

        import threading

        threading.Thread(target=_run, daemon=True).start()

    @QtCore.pyqtSlot(str)
    def _on_rephrase_done(self, result: str) -> None:
        self._rephrase_btn.setText("Rephrase")
        self._rephrase_btn.setEnabled(True)
        if not result.strip():
            show_toast("Rephrase failed", color="#E53935")
            return
        self._show_preview(result, "Rephrased version:")
        show_toast("Rephrase ready — use Copy to copy", color="#1565C0")

    def _show_preview(self, text: str, label: str = "Result:") -> None:
        self._preview_copy_text = text or ""
        self._preview_title.setText(label)
        self._preview_content.setText((text or "")[:300])
        self._preview_box.show()
        cur = int(self._preview_box.maximumHeight())
        if cur <= 0:
            self._preview_box.setMaximumHeight(0)
            self._animate_preview_height(120)
        else:
            self._preview_box.setMaximumHeight(120)

    def _animate_preview_height(self, end: int) -> None:
        if self._preview_height_anim is not None:
            self._preview_height_anim.stop()
            self._preview_height_anim.deleteLater()
            self._preview_height_anim = None
        start = int(self._preview_box.maximumHeight())
        if start == end:
            if end <= 0:
                self._preview_box.hide()
            return
        anim = QtCore.QPropertyAnimation(self._preview_box, b"maximumHeight", self)
        anim.setDuration(150)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._preview_height_anim = anim
        anim.finished.connect(self._on_preview_height_anim_finished)
        anim.start()

    def _on_preview_height_anim_finished(self) -> None:
        self._preview_height_anim = None
        if int(self._preview_box.maximumHeight()) <= 0:
            self._preview_box.hide()

    def _hide_preview_animated(self) -> None:
        if not self._preview_box.isVisible() and int(self._preview_box.maximumHeight()) <= 0:
            return
        self._animate_preview_height(0)

    def _on_copy_preview_clicked(self) -> None:
        self._copy_preview(self._preview_copy_text)

    def _copy_preview(self, text: str) -> None:
        try:
            import pyperclip

            pyperclip.copy(text)
        except Exception:
            QtGui.QGuiApplication.clipboard().setText(text)
        show_toast("Copied to clipboard", color="#02C39A", parent=self)
        self._hide_preview_animated()

    def _on_fix_all_clicked(self) -> None:
        todo = [c for c in self._issue_cards if c.is_actionable()]
        if not todo or self._fix_all_running:
            return
        self._fix_all_running = True
        self._fix_all_queue = todo
        self._fix_all_idx = 0
        self._set_footer_actions_enabled(False)
        self._run_fix_all_step()

    def _run_fix_all_step(self) -> None:
        if not self._fix_all_running:
            return
        if self._fix_all_idx >= len(self._fix_all_queue):
            self._finish_fix_all_done()
            return
        i = self._fix_all_idx
        n = len(self._fix_all_queue)
        self._lbl_issue_count.setText(f"Fixing {i + 1} of {n}…")
        self._lbl_issue_count.setStyleSheet(self._issue_count_default_style)
        card = self._fix_all_queue[i]
        card.set_fixing()
        QtCore.QTimer.singleShot(400, self._finish_one_fix_all)

    def _finish_one_fix_all(self) -> None:
        if not self._fix_all_running:
            return
        if self._fix_all_idx >= len(self._fix_all_queue):
            return
        card = self._fix_all_queue[self._fix_all_idx]
        if not card._fixing_lbl.isVisible():
            return
        if card._span is not None:
            self._fixed_spans.append(card._span)
            self._clipboard_from_fixed_spans()
        else:
            self._clipboard_after_trigger_fix()
        card.set_fixed()
        self._fix_all_idx += 1
        if self._fix_all_idx >= len(self._fix_all_queue):
            self._finish_fix_all_done()
            return
        QtCore.QTimer.singleShot(200, self._run_fix_all_step)

    def _finish_fix_all_done(self) -> None:
        self._fix_all_running = False
        self._set_footer_actions_enabled(True)
        self._update_issue_count()
        if self._fixed_spans:
            fixed_text = redact_for_clipboard(
                self._original_text,
                self._fixed_spans,
                mode=self._current_mask_mode(),
            )
        else:
            fixed_text = remediate_text(self._original_text, "mask")
        if self._hold_mode:
            self._hold_complete_safe_flow(
                fixed_text,
                "Fixed — safe version ready to paste",
            )
            return
        show_toast("All issues fixed", color="#02C39A")
        self._show_preview(fixed_text, "Fixed version:")

    def _show_snooze_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        for minutes, label, toast_lbl in (
            (5.0, "Snooze 5 minutes", "5 minutes"),
            (10.0, "Snooze 10 minutes", "10 minutes"),
            (30.0, "Snooze 30 minutes", "30 minutes"),
            (60.0, "Snooze 1 hour", "1 hour"),
        ):
            act = menu.addAction(label)
            act.triggered.connect(
                lambda _checked=False, m=minutes, lbl=toast_lbl: self._do_snooze(m, lbl)
            )
        gp = self._snooze_drop_btn.mapToGlobal(self._snooze_drop_btn.rect().topLeft())
        menu.adjustSize()
        gh = menu.sizeHint().height()
        menu.exec(QtCore.QPoint(gp.x(), gp.y() - gh))

    def _redact_and_copy(self) -> None:
        self._on_redact_all()

    def _rephrase_and_copy(self) -> None:
        self._on_rephrase()

    def _do_snooze(self, minutes: float, label: str) -> None:
        snooze_guard_minutes(minutes)
        show_toast(f"Snoozed for {label}", color="#F57F17", duration=3000)
        self.slide_out_and_hide()

    def _proceed_without_changes(self) -> None:
        try:
            rs = self._result.get("risk_score", 0)
            try:
                conf = max(0.0, min(1.0, float(rs) / 100.0))
            except (TypeError, ValueError):
                conf = 0.0
            _log_scoring_event(
                self._original_text or "",
                pii_class=f"proceed_anyway:{self._result.get('risk', 'unknown')}",
                confidence=conf,
                action="proceed_anyway",
                context={"source": "remediation_dialog"},
            )
            log_remediation_event(
                "proceed_without_changes",
                {
                    "risk": str(self._result.get("risk", "unknown")),
                    "risk_score": self._result.get("risk_score"),
                },
                text=self._original_text or "",
            )
        except Exception:
            pass
        self._slide_out_pending_accept = True
        self.slide_out_and_hide()

    def _install_app_event_filter(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None or self._app_filter_installed:
            return
        app.installEventFilter(self)
        self._app_filter_installed = True

    def _remove_app_event_filter(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None or not self._app_filter_installed:
            return
        app.removeEventFilter(self)
        self._app_filter_installed = False

    def _should_close_on_outside_click(self, global_pos: QtCore.QPoint) -> bool:
        if self.frameGeometry().contains(global_pos):
            return False
        ap = QtWidgets.QApplication.activePopupWidget()
        if ap is not None:
            return False
        w = QtWidgets.QApplication.widgetAt(global_pos)
        p = w
        while p is not None:
            if p is self:
                return False
            if isinstance(p, Toast):
                return False
            if isinstance(p, (QtWidgets.QMenu, QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox)):
                return False
            p = p.parentWidget()
        return True

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if not self.isVisible():
            return super().eventFilter(obj, event)
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            me = event
            if isinstance(me, QtGui.QMouseEvent):
                gp = me.globalPosition().toPoint()
                if self._should_close_on_outside_click(gp):
                    self.slide_out_and_hide()
            return False
        return super().eventFilter(obj, event)

    def _refresh_power_style(self) -> None:
        if is_monitoring_paused():
            self._power_btn.setStyleSheet(
                "QToolButton { border: none; font-size: 16px; background: transparent; color: #6e6e72; }"
                "QToolButton:hover { color: #9e9ea2; }"
            )
            self._power_btn.setToolTip("Monitoring paused")
        else:
            self._power_btn.setStyleSheet(
                "QToolButton { border: none; font-size: 16px; background: transparent; color: #02C39A; }"
                "QToolButton:hover { color: #3dd4b0; }"
            )
            self._power_btn.setToolTip("Monitoring active — click to pause")

    def _toggle_monitoring_pause(self) -> None:
        set_monitoring_paused(not is_monitoring_paused())
        paused = is_monitoring_paused()
        self._refresh_power_style()
        self.monitoring_toggled.emit(paused)
        if paused:
            par = self.parent()
            if par is not None and hasattr(par, "set_idle"):
                par.set_idle()

    def _screen_geom(self) -> QtCore.QRect:
        screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1920, 1080)

    def _end_pos(self) -> QtCore.QPoint:
        g = self._screen_geom()
        x = g.x() + g.width() - PANEL_W
        return QtCore.QPoint(x, g.y())

    def _start_pos_off_right(self) -> QtCore.QPoint:
        g = self._screen_geom()
        return QtCore.QPoint(g.x() + g.width(), g.y())

    def _save_panel_position(self) -> None:
        try:
            d = user_settings.load()
            d["panel_x"] = self.x()
            d["panel_y"] = self.y()
            user_settings.save(d)
        except Exception:
            pass

    def _restore_panel_position(self) -> None:
        try:
            s = user_settings.load()
            x = s.get("panel_x")
            y = s.get("panel_y")
            if x is None or y is None:
                return
            app = QtWidgets.QApplication.instance()
            ps = app.primaryScreen() if app else None
            screen = ps.availableGeometry() if ps else QtCore.QRect(0, 0, 1920, 1080)
            w = max(self.width(), PANEL_W)
            h = self.height()
            x = max(screen.left(), min(int(x), screen.right() - w))
            y = max(screen.top(), min(int(y), screen.bottom() - h))
            self.move(x, y)
        except Exception:
            pass

    def _header_mouse_press(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        QtWidgets.QWidget.mousePressEvent(self._header_widget, event)

    def _header_mouse_move(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging and self._drag_offset is not None:
            new_pos = event.globalPosition().toPoint() - self._drag_offset
            app = QtWidgets.QApplication.instance()
            ps = app.primaryScreen() if app else None
            screen = ps.availableGeometry() if ps else QtCore.QRect(0, 0, 1920, 1080)
            w = max(self.width(), PANEL_W)
            h = self.height()
            new_pos.setX(max(screen.left(), min(new_pos.x(), screen.right() - w)))
            new_pos.setY(max(screen.top(), min(new_pos.y(), screen.bottom() - h)))
            self.move(new_pos)
            event.accept()
            return
        QtWidgets.QWidget.mouseMoveEvent(self._header_widget, event)

    def _header_mouse_release(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_offset = None
            self.unsetCursor()
            self._header_widget.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            self._save_panel_position()
            event.accept()
            return
        QtWidgets.QWidget.mouseReleaseEvent(self._header_widget, event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        self._apply_panel_height()
        if not self._slide_in_done:
            bubble = self._bubble
            app = QtWidgets.QApplication.instance()
            scr = (
                QtGui.QGuiApplication.screenAt(bubble.pos())
                if bubble is not None
                else (app.primaryScreen() if app else None)
            )
            if scr is None and app:
                scr = app.primaryScreen()
            screen = scr.availableGeometry() if scr else QtCore.QRect(0, 0, 1920, 1080)

            saved = user_settings.load()
            px, py = saved.get("panel_x"), saved.get("panel_y")
            has_saved = px is not None and py is not None

            if has_saved:
                w = max(self.width(), PANEL_W)
                h = self.height()
                ex = max(screen.left(), min(int(px), screen.right() - w))
                ey = max(screen.top(), min(int(py), screen.bottom() - h))
                end = QtCore.QPoint(ex, ey)
                self._pending_end_pos = end
                self.move(QtCore.QPoint(end.x() + 80, end.y()))
            elif bubble is not None:
                if bubble.x() + bubble.width() + PANEL_W + 8 < screen.right():
                    panel_x = bubble.x() + bubble.width() + 8
                else:
                    panel_x = bubble.x() - PANEL_W - 8
                panel_y = max(
                    screen.top(),
                    min(bubble.y(), screen.bottom() - self.height()),
                )
                end = QtCore.QPoint(int(panel_x), int(panel_y))
                self._pending_end_pos = end
                self.move(QtCore.QPoint(end.x() + 80, end.y()))
            else:
                self.move(self._start_pos_off_right())
                self._pending_end_pos = self._end_pos()
            self._slide_in_done = True
        super().showEvent(event)
        if self._pending_end_pos is not None:
            end = self._pending_end_pos
            self._pending_end_pos = None
            QtCore.QTimer.singleShot(0, lambda: self._animate_pos_to(end))
        QtCore.QTimer.singleShot(0, self._install_app_event_filter)
        QtCore.QTimer.singleShot(ANIM_MS + 10, self._restore_panel_position)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self._remove_app_event_filter()
        super().hideEvent(event)

    def _animate_pos_to(self, end_pos: QtCore.QPoint) -> None:
        self._pos_anim = QtCore.QPropertyAnimation(self, b"pos", self)
        self._pos_anim.setDuration(ANIM_MS)
        self._pos_anim.setStartValue(self.pos())
        self._pos_anim.setEndValue(end_pos)
        self._pos_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._pos_anim.start()

    def slide_out_and_hide(self) -> None:
        if self._closing_anim:
            return
        self._closing_anim = True
        g = self._screen_geom()
        end = QtCore.QPoint(g.x() + g.width(), self.y())
        anim = QtCore.QPropertyAnimation(self, b"pos", self)
        anim.setDuration(ANIM_MS)
        anim.setStartValue(self.pos())
        anim.setEndValue(end)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        anim.finished.connect(self._after_slide_out_hide)
        anim.start()

    def _after_slide_out_hide(self) -> None:
        self._remove_app_event_filter()
        self._closing_anim = False
        self.hide()
        self._slide_in_done = False
        pending = self._slide_out_pending_accept
        self._slide_out_pending_accept = False
        if pending:
            super().accept()
        else:
            self.remediation_finished.emit(False)

    def accept(self) -> None:
        super().accept()

    def reject(self) -> None:
        super().reject()

