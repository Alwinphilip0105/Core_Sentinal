"""
Full-height right-edge drawer: Sentinel remediation panel (non-modal).
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6 import QtCore, QtGui, QtWidgets

import user_settings
from guardrail_runtime import is_monitoring_paused, set_monitoring_paused, snooze_guard_minutes
from infer import filter_triggers_already_in_spans
from guardrail_logs import log_remediation_event
from infer import _log_scoring_event
from pii_remediation import (
    _resolve_span_bounds,
    mask_pii_spans,
    remediate_text,
    rephrase_text,
)
from toast import Toast, show_toast
from font_clamp import MIN_PX_BODY, paint_font_px

FONT_FAMILY = "Segoe UI"
PANEL_W = 408
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


def _strip_risk_colors(span_or_trigger: dict | str) -> tuple[QtGui.QColor, str]:
    if isinstance(span_or_trigger, dict):
        r = str(span_or_trigger.get("risk", "low")).lower()
        if r in ("high", "h"):
            return QtGui.QColor("#E53935"), "high"
        if r in ("med", "medium", "m"):
            return QtGui.QColor("#F9A825"), "med"
        return QtGui.QColor("#F9A825"), "med"


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
        col, _, _, _ = _score_tier(self._score)
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
    ):
        super().__init__(parent)
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
        triggers = filter_triggers_already_in_spans(
            [s for s in spans if isinstance(s, dict)], list(triggers)
        )
        issue_count = len(spans) + len(triggers)
        if issue_count == 0:
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

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

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

        if not self._has_issues:
            empty_wrap = QtWidgets.QWidget()
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
            mlay = QtWidgets.QHBoxLayout(mask_bar)
            mlay.setContentsMargins(12, 6, 12, 4)
            ml = QtWidgets.QLabel("Mask style")
            ml.setStyleSheet("font-size: 11px; color: #6b7280; font-family: Segoe UI;")
            self._combo_mask_mode = QtWidgets.QComboBox()
            self._combo_mask_mode.addItems(
                ["Partial (smart)", "Full replace", "Stars", "Type only"]
            )
            self._combo_mask_mode.setStyleSheet(_PANEL_COMBOBOX_QSS)
            self._combo_mask_mode.setCurrentIndex(0)
            self._combo_mask_mode.currentIndexChanged.connect(self._on_mask_mode_changed)
            mlay.addWidget(ml)
            mlay.addStretch()
            mlay.addWidget(self._combo_mask_mode)
            cv.addWidget(mask_bar)

            first = True
            for s in spans:
                if not isinstance(s, dict):
                    continue
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

        fl.addWidget(footer_row)

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

        self.setFixedWidth(PANEL_W)
        self._apply_panel_height()
        self._update_issue_count()

    def _apply_panel_height(self) -> None:
        screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        g = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1080, 1920)
        self.setFixedHeight(g.height())

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
        if not self._has_issues or not self._issue_cards:
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

    def _on_mask_mode_changed(self, _index: int = 0) -> None:
        keys = ("partial", "full", "stars", "type_only")
        if not hasattr(self, "_combo_mask_mode"):
            return
        i = self._combo_mask_mode.currentIndex()
        self._mask_mode = keys[i] if 0 <= i < len(keys) else "partial"

    def _current_mask_mode(self) -> str:
        if hasattr(self, "_combo_mask_mode"):
            keys = ("partial", "full", "stars", "type_only")
            i = self._combo_mask_mode.currentIndex()
            return keys[i] if 0 <= i < len(keys) else "partial"
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
        show_toast("Fixed — safe version copied", color="#02C39A")

    def _on_card_skip(self, card: _IssueCardWidget) -> None:
        if not card.is_actionable() or self._fix_all_running:
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
        show_toast("All issues fixed", color="#02C39A")
        if self._fixed_spans:
            fixed_text = mask_pii_spans(
                self._original_text,
                self._fixed_spans,
                mode=self._current_mask_mode(),
            )
        else:
            fixed_text = remediate_text(self._original_text, "mask")
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

