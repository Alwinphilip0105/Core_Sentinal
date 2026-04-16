"""
Grammarly-style floating pill for Core Sentinel guardrail.
Single window: traffic light + pill (toolbar icons dock to the right of the bubble as a child widget).
"""

from __future__ import annotations

import datetime
import importlib.util
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6 import QtCore, QtGui, QtWidgets

from active_window_llm import detect_llm_window, get_active_llm_name
from guardrail_runtime import get_monitor_llm_only, is_monitoring_paused, set_monitoring_paused
from pii_remediation import encrypt_pii, encrypt_text, hash_pii, mask_pii, mask_pii_spans, redact_all_literal

import user_settings
from character_widget import CharacterWidget
from infer import score_clipboard_with_pii
from toast import show_toast
from traffic_light_indicator import COLOR_HOUSING, H_TL, TrafficLightIndicator, W_TL
from ui_bubble_toolbar import BubbleToolbar, ToolbarTooltip
from font_clamp import MIN_PX_BADGE, MIN_PX_BODY, MIN_PX_LABEL, paint_font_px

# Traffic column is embedded at x=0; pill starts after it (single top-level window).
TRAFFIC_SLOT_W = W_TL
# Outer widget height; width animates (collapsed / expanded pill)
COLLAPSED_W = 72 + TRAFFIC_SLOT_W
EXPANDED_W = 200 + TRAFFIC_SLOT_W
EXPANDED_W_NO_STREAK = 180 + TRAFFIC_SLOT_W
PILL_H = 34
WIDGET_H = max(46, H_TL)
# Pill geometry inside widget (power + shield + score); width = widget - PILL_X - PILL_MARGIN_RIGHT
# Start exactly at traffic-slot edge so the two parts are fused without a visible gap.
PILL_X = TRAFFIC_SLOT_W
PILL_Y = max(6, (WIDGET_H - PILL_H) // 2)
PILL_MARGIN_RIGHT = 22
PILL_RX = 14

# Vertical toolbar docked to the right of the pill row (child widget; expands window width).
TOOLBAR_DOCK_MARGIN = 4
TOOLBAR_DOCK_BODY_W = 40
ALWAYS_SHOW_TOOLBAR = False

def _pill_body_path(pill_w: float) -> QtGui.QPainterPath:
    """Pill fill: square left edge (flush with traffic light), rounded right corners only."""
    x = float(PILL_X)
    y = float(PILL_Y)
    w = float(pill_w)
    h = float(PILL_H)
    r = float(PILL_RX)
    path = QtGui.QPainterPath()
    path.moveTo(x, y)
    path.lineTo(x + w - r, y)
    path.arcTo(QtCore.QRectF(x + w - 2.0 * r, y, 2.0 * r, 2.0 * r), 90.0, -90.0)
    path.lineTo(x + w, y + h - r)
    path.arcTo(QtCore.QRectF(x + w - 2.0 * r, y + h - 2.0 * r, 2.0 * r, 2.0 * r), 0.0, -90.0)
    path.lineTo(x, y + h)
    path.closeSubpath()
    return path


CARD_W = 280
CARD_SLIDE_PX = 36

COLOR_PILL_PAUSED = QtGui.QColor(88, 88, 90)
COLOR_SHIELD_OFF = QtGui.QColor(0x96, 0x98, 0x9A)
COLOR_SHIELD_ON = QtGui.QColor(0x02, 0xC3, 0x9A)
COLOR_SPINNER = QtGui.QColor(0x02, 0xC3, 0x9A)
COLOR_CHECK = QtGui.QColor(0x02, 0xC3, 0x9A)
COLOR_CRITICAL_COUNT = QtGui.QColor(0xFF, 0x8A, 0x80)
COLOR_IDLE_GLYPH = QtGui.QColor(0x02, 0xC3, 0x9A)
CARD_QSS = (
    "QFrame#actionCard { background: rgba(26,26,26,230); border: 1px solid #3a3a3a; "
    "border-radius: 10px; }"
    "QLabel { color: #d0d0d0; font-family: 'Segoe UI'; font-size: 11px; }"
    "QPlainTextEdit { background: #141414; color: #e8e8e8; border: 1px solid #333; "
    "border-radius: 6px; padding: 6px; font-family: 'Segoe UI'; font-size: 10px; }"
    "QPushButton { font-family: 'Segoe UI'; font-size: 11px; padding: 6px 12px; "
    "border-radius: 6px; background: #3d3d3d; color: #eee; border: 1px solid #555; }"
    "QPushButton:hover { background: #4a4a4a; }"
    "QPushButton#primary { background: #0d6efd; border-color: #0d6efd; color: white; }"
    "QPushButton#primary:hover { background: #0b5ed7; }"
)


def _derive_critical_task_count(result: dict) -> int:
    """
    Items that should show as “critical” in the pill: API/secret hits, high risk, or strong scores.
    """
    if not isinstance(result, dict):
        return 0
    if bool(result.get("critical_secret_detected")):
        spans = result.get("spans") if isinstance(result.get("spans"), list) else []
        return max(len(spans), 1)
    risk = str(result.get("risk", "low")).lower()
    try:
        score = int(result.get("risk_score", 0) or 0)
    except (TypeError, ValueError):
        score = 0
    spans = result.get("spans") if isinstance(result.get("spans"), list) else []
    triggers = result.get("triggers") or []
    if not isinstance(triggers, (list, tuple)):
        triggers = []
    if risk == "high":
        return max(len(spans), 1) if spans else 1
    if score >= 75:
        return max(len(spans), len(triggers), 1)
    action = str(result.get("action", "") or "")
    if action == "block" and score >= 55:
        return max(len(spans), len(triggers), 1)
    return 0


class CharacterEvent(QtWidgets.QWidget):
    """Ephemeral celebration / info popup near the pill."""

    def __init__(
        self,
        emoji: str,
        title: str,
        subtitle: str,
        color: str,
        duration: int = 4000,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setFixedWidth(160)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        emoji_lbl = QtWidgets.QLabel(emoji)
        emoji_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        emoji_lbl.setStyleSheet("font-size: 32px; background: transparent;")
        layout.addWidget(emoji_lbl)

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        title_lbl.setWordWrap(True)
        layout.addWidget(title_lbl)

        sub_lbl = QtWidgets.QLabel(subtitle)
        sub_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sub_lbl.setStyleSheet(
            "color: rgba(255,255,255,0.5); font-size: 10px; background: transparent;"
        )
        layout.addWidget(sub_lbl)

        self.adjustSize()
        self.setStyleSheet(
            "QWidget { background: rgba(22,22,24,235); border-radius: 12px; "
            "border: 1px solid rgba(255,255,255,0.1); }"
        )

        self._eff = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._eff)

        self._in_anim = QtCore.QPropertyAnimation(self._eff, b"opacity", self)
        self._in_anim.setDuration(250)
        self._in_anim.setStartValue(0.0)
        self._in_anim.setEndValue(1.0)
        self._in_anim.start()

        QtCore.QTimer.singleShot(duration, self._dismiss)
        self.show()
        self.raise_()

    def _dismiss(self) -> None:
        self._out_anim = QtCore.QPropertyAnimation(self._eff, b"opacity", self)
        self._out_anim.setDuration(300)
        self._out_anim.setStartValue(1.0)
        self._out_anim.setEndValue(0.0)
        self._out_anim.finished.connect(self.close)
        self._out_anim.start()

    def _position_near_bubble(self, bubble: QtWidgets.QWidget) -> None:
        bg = bubble.mapToGlobal(QtCore.QPoint(0, 0))
        screen = QtWidgets.QApplication.primaryScreen()
        g = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1920, 1080)
        x = bg.x() + bubble.width() // 2 - self.width() // 2
        y = bg.y() - self.height() - 10
        if y < g.top() + 4:
            y = bg.y() + bubble.height() + 10
        x = max(g.left() + 4, min(x, g.right() - self.width() - 4))
        self.move(x, y)


class RiskBubble(QtWidgets.QWidget):
    """
    Grammarly-like pill: paint-only chrome; optional action card stack above; user-draggable position.
    """

    STREAK_MILESTONES = {
        5: ("😊", "5 safe pastes!", "#43A047"),
        10: ("🎉", "10 in a row!", "#43A047"),
        25: ("🔥", "25 streak — on fire!", "#FFB300"),
        50: ("⭐", "50 streak — legend!", "#FFB300"),
        100: ("🏆", "100 streak — master!", "#02C39A"),
    }

    menu_action_clicked = QtCore.pyqtSignal(str)
    context_menu_action = QtCore.pyqtSignal(str)
    monitoring_pause_changed = QtCore.pyqtSignal(bool)
    document_scan_progress = QtCore.pyqtSignal(int, str)
    document_scan_finished = QtCore.pyqtSignal(object)
    document_scan_failed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: dict = {}
        self._llm_target = ""
        self._original_text = ""
        self._url = ""
        self._cleaned_title = ""
        self._dragging = False
        self._drag_offset = QtCore.QPoint(0, 0)
        self._press_pos: Optional[QtCore.QPoint] = None
        self._click_timer = QtCore.QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(280)
        self._click_timer.timeout.connect(self._on_delayed_single_click)

        self._in_llm = False
        self._analysing = False
        self._hover_tooltip_visible = False
        self._issue_count = 0
        self._show_all_clear = False
        self._card_anim: Optional[QtCore.QPropertyAnimation] = None
        self._action_card_inner: Optional[QtWidgets.QWidget] = None
        self._redact_toast_timer = QtCore.QTimer(self)
        self._redact_toast_timer.setSingleShot(True)
        self._redact_toast_timer.timeout.connect(self._dismiss_action_card)

        self._last_text = ""
        self._last_result: dict = {}
        self._last_label = ""
        self._feedback_widget: Optional[QtWidgets.QWidget] = None
        self._overlay_feedback_prompt: Optional[QtWidgets.QWidget] = None
        self._feedback_prompt: Optional[QtWidgets.QWidget] = None
        self._open_panel_callback: Optional[object] = None
        self._feedback_timer: Optional[QtCore.QTimer] = None
        self._feedback_survey_active = False
        self._feedback_survey_timer: Optional[QtCore.QTimer] = None
        self._action_card_reuse: Optional[QtWidgets.QFrame] = None

        self._spin_angle = 0.0
        self._spinner_opacity = 0.0

        self._show_badge_on_issues = True

        self._risk_score = 0
        self._has_risk_score = False
        # Secrets / high-risk / block-worthy items — shown in score slot; dims ambient opacity
        self._critical_task_count = 0
        self._idle_pixmap: Optional[QtGui.QPixmap] = None
        self._idle_pixmap_source: str = ""
        # Keep default idle chrome transparent (remove persistent background frame).
        self._current_bg = QtGui.QColor(0, 0, 0, 0)
        # Background clipboard monitor preview (not a live paste)
        self._clipboard_preview_active = False
        self._preview_level = "safe"
        self._preview_score = 0
        self._preview_bg = QtGui.QColor(28, 28, 30, 0)

        self._llm_name = ""
        self._last_action = "silent"
        self._last_score = 0
        self._char_event: Optional[QtWidgets.QWidget] = None
        self._milestone_widget: Optional[QtWidgets.QWidget] = None
        self._streak = self._load_streak()

        self._current_width = float(COLLAPSED_W)
        self._target_width = COLLAPSED_W
        self._expand_timer = QtCore.QTimer(self)
        self._expand_timer.setInterval(16)
        self._expand_timer.timeout.connect(self._tick_expand)

        self._displayed_score = 0.0
        self._target_score = 0
        self._ring_timer = QtCore.QTimer(self)
        self._ring_timer.setInterval(33)
        self._ring_timer.timeout.connect(self._tick_ring)

        self._hold_locked = False
        self._hold_critical = False
        self._hold_pulse = True
        self._hold_timer: Optional[QtCore.QTimer] = None

        self._spinner_timer = QtCore.QTimer(self)
        self._spinner_timer.setInterval(120)
        self._spinner_timer.timeout.connect(self._tick_spinner)

        self._all_clear_timer = QtCore.QTimer(self)
        self._all_clear_timer.setSingleShot(True)
        self._all_clear_timer.timeout.connect(self._hide_all_clear_indicator)

        self._remediation_panel: Optional[QtWidgets.QWidget] = None

        self._dot_opacity = 1.0
        self._dot_pulse_timer = QtCore.QTimer(self)
        self._dot_pulse_timer.setInterval(800)
        self._dot_pulse_timer.timeout.connect(self._tick_dot_pulse)

        self._toolbar: Optional[BubbleToolbar] = None
        self._tb_tooltip = ToolbarTooltip()
        self._toolbar_hide_timer = QtCore.QTimer(self)
        self._toolbar_hide_timer.setSingleShot(True)
        self._toolbar_hide_timer.setInterval(400)
        self._toolbar_hide_timer.timeout.connect(self._maybe_hide_toolbar)
        self._toolbar_idle_timer = QtCore.QTimer(self)
        self._toolbar_idle_timer.setSingleShot(True)
        self._toolbar_idle_timer.setInterval(4000)
        self._toolbar_idle_timer.timeout.connect(self._hide_toolbar_idle_timeout)
        self._status_hide_timer: Optional[QtCore.QTimer] = None

        ga = QtGui.QGuiApplication.instance()
        if ga is not None:
            ga.applicationStateChanged.connect(self._on_application_state_changed)
        self._toolbar_anim: Optional[QtCore.QPropertyAnimation] = None

        self._is_hovered = False
        self._target_opacity = 0.72
        self._opacity_timer = QtCore.QTimer(self)
        self._opacity_timer.setInterval(40)
        self._opacity_timer.timeout.connect(self._tick_opacity)
        self._last_monitor_state: tuple[bool, bool, str, bool] | None = None
        self._last_monitor_refresh_mono = 0.0
        self._cached_idle_badge: QtGui.QPixmap | None = None

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))

        self._action_card_wrap = QtWidgets.QWidget(self)
        self._action_card_wrap.setFixedHeight(0)
        self._action_card_wrap.setFixedWidth(CARD_W)

        self._tooltip_win: Optional[QtWidgets.QLabel] = None

        self._character_widget: Optional[CharacterWidget] = None
        self._character_first_scored_done = False

        _pw0 = COLLAPSED_W - PILL_X - PILL_MARGIN_RIGHT
        self._traffic_indicator = TrafficLightIndicator(
            self, QtCore.QRect(PILL_X, PILL_Y, _pw0, PILL_H)
        )
        self._traffic_indicator.remediation_requested.connect(self._on_indicator_remediation_click)

        self.setFixedSize(COLLAPSED_W, WIDGET_H)

        self._restore_pill_position()

        self._monitor_timer = QtCore.QTimer(self)
        self._monitor_timer.timeout.connect(self._refresh_monitoring)
        self._monitor_timer.setInterval(3500)
        self._monitor_timer.start()

        self._refresh_monitoring()
        self.setWindowOpacity(self._ambient_opacity())
        self._sync_traffic_indicator_opacity()
        self._opacity_timer.stop()

        self.document_scan_progress.connect(self._on_scan_progress)
        self.document_scan_finished.connect(self._on_scan_done)
        self.document_scan_failed.connect(self._on_scan_failed)

        self._open_panel_callback = self._on_delayed_single_click

    @property
    def _state(self) -> str:
        return str(getattr(self._traffic_indicator, "_state", "") or "")

    def _content_width_for_layout(self) -> int:
        """Pill row width from expand animation (excludes toolbar dock slot)."""
        cw = int(round(self._current_width))
        if self._action_card_wrap.height() > 0:
            return max(CARD_W, cw)
        return cw

    def _toolbar_slot_w(self) -> int:
        # Toolbar is now a separate top-level tool window; no in-pill slot reservation.
        return 0

    def _pill_row_top(self) -> int:
        return self._action_card_wrap.height()

    def _pill_width_f(self) -> float:
        base = float(self._content_width_for_layout())
        w = base - float(PILL_X) - float(PILL_MARGIN_RIGHT)
        return max(float(PILL_RX * 2 + 4), w)

    def _refresh_window_geometry(self, *, reserve_toolbar_slot: bool = False) -> None:
        h = self._action_card_wrap.height() + WIDGET_H
        slot = self._toolbar_slot_w()
        # Toolbar is a separate top-level window; never reserve extra pill width for it.
        del reserve_toolbar_slot
        w = self._content_width_for_layout() + slot
        self.setFixedSize(w, max(h, WIDGET_H))

    def _apply_window_width_from_expand(self) -> None:
        self._refresh_window_geometry()

    def _tick_expand(self) -> None:
        diff = float(self._target_width) - self._current_width
        if abs(diff) < 0.5:
            self._current_width = float(self._target_width)
            self._expand_timer.stop()
        else:
            self._current_width += diff * 0.18
        self._apply_window_width_from_expand()
        self._reposition_traffic_light()
        if self._toolbar is not None and self._toolbar.isVisible():
            self._update_toolbar_position()
        self.update()

    def _reposition_traffic_light(self) -> None:
        pw = max(PILL_RX * 2 + 1, int(self._pill_width_f()))
        self._traffic_indicator.set_pill_rect(QtCore.QRect(PILL_X, PILL_Y, pw, PILL_H))

    def _load_streak(self) -> dict:
        d = user_settings.load()
        last = str(d.get("streak_last_date") or "")
        try:
            if last:
                ld = datetime.date.fromisoformat(last[:10])
                cutoff = datetime.date.today() - datetime.timedelta(days=2)
                if ld < cutoff:
                    d["streak_count"] = 0
                    user_settings.save(d)
        except Exception:
            pass
        d = user_settings.load()
        return {
            "streak_count": int(d.get("streak_count", 0) or 0),
            "streak_best": int(d.get("streak_best", 0) or 0),
            "streak_last_date": str(d.get("streak_last_date") or ""),
            "total_safe": int(d.get("total_safe", 0) or 0),
            "total_risky": int(d.get("total_risky", 0) or 0),
        }

    def _save_streak(self) -> None:
        d = user_settings.load()
        d["streak_count"] = int(self._streak.get("streak_count", 0))
        d["streak_best"] = int(self._streak.get("streak_best", 0))
        d["streak_last_date"] = str(self._streak.get("streak_last_date") or "")
        d["total_safe"] = int(self._streak.get("total_safe", 0))
        d["total_risky"] = int(self._streak.get("total_risky", 0))
        user_settings.save(d)

    def _show_streak_milestone(self, count: int) -> None:
        msgs = {
            5: ("🎉", "5 safe in a row!", "Great habits forming!", "#43A047"),
            10: ("🔥", "10 streak!", "You are on fire!", "#FFB300"),
            25: ("⭐", "25 safe pastes!", "Incredible discipline!", "#FFB300"),
            50: ("🏆", "50 streak!", "Legend status!", "#02C39A"),
            100: ("👑", "100 in a row!", "Absolute master!", "#9C27B0"),
        }
        emoji, title, subtitle, color = msgs.get(
            count,
            ("✓", f"{count} safe!", "Keep it up!", "#43A047"),
        )
        w = CharacterEvent(
            emoji=emoji,
            title=title,
            subtitle=subtitle,
            color=color,
            duration=5000,
        )
        w._position_near_bubble(self)
        self._milestone_widget = w
        show_toast(
            f"{emoji} {title} {subtitle}",
            color=color,
            duration=4000,
            parent=self,
        )

    def _update_streak_from_action(self, action: str) -> None:
        """
        Call from main for every scored paste result so streak state stays in sync
        for silent / warn / block (including when the bubble only does a minimal UI path).
        """
        today = datetime.date.today().isoformat()
        act = str(action)
        if act == "silent":
            self._streak["streak_count"] = int(self._streak.get("streak_count", 0)) + 1
            self._streak["streak_last_date"] = today
            self._streak["total_safe"] = int(self._streak.get("total_safe", 0)) + 1
            if self._streak["streak_count"] > int(self._streak.get("streak_best", 0)):
                self._streak["streak_best"] = self._streak["streak_count"]
            count = self._streak["streak_count"]
            print(f"[streak] safe paste #{count}", flush=True)
            milestones = {5, 10, 25, 50, 100}
            if count in milestones:
                print(f"[streak] MILESTONE {count}!", flush=True)
                self._show_streak_milestone(count)
        elif act in ("warn", "block"):
            old = int(self._streak.get("streak_count", 0))
            self._streak["streak_count"] = 0
            self._streak["total_risky"] = int(self._streak.get("total_risky", 0)) + 1
            print(f"[streak] reset (was {old})", flush=True)
            if old >= 3:
                self._show_streak_broken(old)
        self._save_streak()
        self.update()

    def _check_milestone(self, count: int) -> None:
        if count in self.STREAK_MILESTONES:
            emoji, msg, color = self.STREAK_MILESTONES[count]
            self._show_streak_celebration(count, emoji, msg, color)

    def _show_streak_celebration(
        self, count: int, emoji: str, msg: str, color: str
    ) -> None:
        self._show_character_event(
            emoji=emoji,
            title=msg,
            subtitle=f"Best: {self._streak['streak_best']}",
            color=color,
            duration=5000,
        )
        show_toast(
            f"{emoji} {msg} Keep it up!",
            color=color,
            duration=4000,
            parent=self,
        )

    def _show_streak_broken(self, old_streak: int) -> None:
        self._show_character_event(
            emoji="😢",
            title=f"Streak of {old_streak} broken",
            subtitle="Start a new one!",
            color="#888888",
            duration=3000,
        )

    def _show_character_event(
        self,
        emoji: str,
        title: str,
        subtitle: str,
        color: str,
        duration: int = 4000,
    ) -> None:
        w = CharacterEvent(emoji, title, subtitle, color, duration)
        w._position_near_bubble(self)
        self._char_event = w

    def _tick_ring(self) -> None:
        diff = float(self._target_score) - self._displayed_score
        if abs(diff) < 1.0:
            self._displayed_score = float(self._target_score)
            self._ring_timer.stop()
        else:
            self._displayed_score += diff * 0.12
        self.update()

    def _should_paint_score_ring(self) -> bool:
        if not self._show_badge_on_issues or self._analysing:
            return False
        if self._risk_score <= 0:
            return False
        if self._issue_count <= 0 and not getattr(self, "_hold_locked", False):
            return False
        st = self._state.lower()
        if st in ("idle", "none", "not_monitoring"):
            return False
        return True

    def _paint_risk_score_ring(self, painter: QtGui.QPainter) -> None:
        if not self._should_paint_score_ring():
            return
        hold = getattr(self, "_hold_locked", False)
        if hold:
            painter.save()
            painter.setOpacity(1.0 if self._hold_pulse else 0.5)
        pill_rect = QtCore.QRectF(
            float(PILL_X), float(PILL_Y), self._pill_width_f(), float(PILL_H)
        )
        ring_size = 28
        ring_x = pill_rect.right() - ring_size + 6.0
        ring_y = pill_rect.top() - 6.0
        cx = ring_x + ring_size / 2.0
        cy = ring_y + ring_size / 2.0
        ring_r = ring_size / 2.0 - 2.0

        ds = max(0.0, min(100.0, self._displayed_score))
        sc = int(round(ds))
        if sc <= 40:
            ring_color = QtGui.QColor("#43A047")
        elif sc <= 70:
            ring_color = QtGui.QColor("#FFB300")
        else:
            ring_color = QtGui.QColor("#E53935")

        track_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 25), 3.0)
        track_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QtCore.QPointF(cx, cy), ring_r, ring_r)

        span = int(ds / 100.0 * 360.0 * 16.0)
        if span > 0:
            arc_pen = QtGui.QPen(ring_color, 3.0)
            arc_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(arc_pen)
            arc_rect = QtCore.QRectF(
                ring_x + 2.0,
                ring_y + 2.0,
                float(ring_size - 4),
                float(ring_size - 4),
            )
            painter.drawArc(arc_rect, 90 * 16, -span)

        fam = self.font().family() or "Segoe UI"
        painter.setPen(ring_color)
        painter.setFont(paint_font_px(fam, 10, bold=True, floor=MIN_PX_LABEL))
        painter.drawText(
            QtCore.QRectF(ring_x, ring_y, float(ring_size), float(ring_size)),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            str(int(round(ds))),
        )
        if self._issue_count > 0:
            painter.setPen(QtGui.QColor(255, 255, 255, 150))
            painter.setFont(paint_font_px(fam, 7, floor=MIN_PX_BADGE))
            painter.drawText(
                QtCore.QRectF(ring_x, cy + 4.0, float(ring_size), 10.0),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"x{self._issue_count}",
            )
        if hold:
            painter.restore()

    def _paint_expanded_hover_content(self, p: QtGui.QPainter, fam: str) -> None:
        st = self._state.lower()
        if st in ("idle", "none", "not_monitoring"):
            self._target_width = COLLAPSED_W
            self._expand_timer.start()
            return
        if self._current_width <= 100:
            return
        expand_pct = max(0.0, (self._current_width - 100.0) / 120.0)
        text_alpha = int(expand_pct * 255.0)
        pill_h = float(PILL_H)
        llm_color = QtGui.QColor(255, 255, 255, text_alpha)
        p.setPen(llm_color)
        p.setFont(paint_font_px(fam, 11, bold=True, floor=MIN_PX_LABEL))
        llm_name = getattr(self, "_llm_name", "") or "Not monitoring"
        p.drawText(
            QtCore.QRectF(64.0, float(PILL_Y), 90.0, pill_h / 2.0),
            QtCore.Qt.AlignmentFlag.AlignVCenter,
            llm_name,
        )
        # SILENT=green, WARN=amber, BLOCK=red
        action_colors = {
            "silent": QtGui.QColor(0x43, 0xA0, 0x47, text_alpha),
            "warn": QtGui.QColor(0xFF, 0xB3, 0x00, text_alpha),
            "block": QtGui.QColor(0xE5, 0x39, 0x35, text_alpha),
        }
        action = str(getattr(self, "_last_action", "silent") or "silent").lower()
        action_color = action_colors.get(
            action, QtGui.QColor(150, 150, 150, text_alpha)
        )
        p.setPen(action_color)
        p.setFont(paint_font_px(fam, 9, floor=MIN_PX_BODY))
        p.drawText(
            QtCore.QRectF(64.0, PILL_Y + pill_h / 2.0 - 2.0, 90.0, pill_h / 2.0),
            QtCore.Qt.AlignmentFlag.AlignVCenter,
            action.upper(),
        )
        count = int(self._streak.get("streak_count", 0) or 0)
        if count > 0:
            if count >= 10:
                streak_color = QtGui.QColor("#FFB300")
                streak_text = f"🔥{count}"
                streak_label = "streak"
            else:
                streak_color = QtGui.QColor("#43A047")
                streak_text = f"✓{count}"
                streak_label = "safe"
            p.setPen(streak_color)
            big_font = QtGui.QFont()
            big_font.setPixelSize(13)
            big_font.setBold(True)
            p.setFont(big_font)
            p.drawText(
                QtCore.QRectF(165.0, 4.0, 50.0, pill_h / 2.0),
                QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
                streak_text,
            )
            p.setPen(QtGui.QColor(255, 255, 255, 80))
            small_font = QtGui.QFont()
            small_font.setPixelSize(8)
            p.setFont(small_font)
            p.drawText(
                QtCore.QRectF(165.0, pill_h / 2.0 - 2.0, 50.0, pill_h / 2.0),
                QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
                streak_label,
            )

    # --- pill center in widget coordinates ---
    def _idle_image_path(self) -> str:
        env = (os.environ.get("GUARDRAIL_BUBBLE_IDLE_IMAGE") or "").strip()
        if env.lower() in ("none", "0", "false", "-", "tick"):
            return ""
        if env:
            return env
        try:
            p = user_settings.load().get("bubble_idle_image")
            if p:
                return str(p).strip()
        except Exception:
            pass
        return ""

    def _get_idle_pixmap(self) -> Optional[QtGui.QPixmap]:
        """Cached pixmap for the idle score slot (PNG/JPG/BMP/GIF; SVG not supported by QPixmap)."""
        raw = self._idle_image_path()
        if not raw:
            self._idle_pixmap = None
            self._idle_pixmap_source = ""
            return None
        path = Path(raw)
        if not path.is_file():
            self._idle_pixmap = None
            return None
        key = str(path.resolve())
        if (
            self._idle_pixmap is not None
            and not self._idle_pixmap.isNull()
            and self._idle_pixmap_source == key
        ):
            return self._idle_pixmap
        pm = QtGui.QPixmap(str(path))
        if pm.isNull():
            self._idle_pixmap = None
            return None
        self._idle_pixmap = pm
        self._idle_pixmap_source = key
        return self._idle_pixmap

    def reload_idle_image(self) -> None:
        """Call after changing user_settings bubble_idle_image so the cache refreshes."""
        self._idle_pixmap = None
        self._idle_pixmap_source = ""
        self.update()

    def _pill_center_local(self) -> QtCore.QPoint:
        pw = int(self._pill_width_f())
        return QtCore.QPoint(PILL_X + pw // 2, PILL_Y + PILL_H // 2)

    def _pill_center_offset_from_window_top(self) -> int:
        return self._action_card_wrap.height() + self._pill_center_local().y()

    def _pill_center_global(self) -> QtCore.QPoint:
        return self.mapToGlobal(self._pill_center_local())

    def _resize_to_state(self) -> None:
        self._refresh_window_geometry()

    def _clamp_point_to_screen(self, top_left: QtCore.QPoint) -> QtCore.QPoint:
        app = QtGui.QGuiApplication.instance()
        screen = app.screenAt(top_left) if app else None
        if screen is None:
            screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return top_left
        g = screen.availableGeometry()
        x = max(g.left(), min(top_left.x(), g.right() - self.width() + 1))
        y = max(g.top(), min(top_left.y(), g.bottom() - self.height() + 1))
        return QtCore.QPoint(x, y)

    def _move_to_default_corner(self) -> None:
        app = QtGui.QGuiApplication.instance()
        screen = app.screenAt(QtGui.QCursor.pos()) if app else None
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if not screen:
            return
        g = screen.availableGeometry()
        self.move(
            g.right() - self.width() - 20,
            g.bottom() - self.height() - 80,
        )

    def _restore_pill_position(self) -> None:
        try:
            d = user_settings.load()
            x = d.get("bubble_x")
            y = d.get("bubble_y")
            if x is not None and y is not None:
                p = QtCore.QPoint(int(x), int(y))
                self.move(self._clamp_point_to_screen(p))
                return
        except Exception:
            pass
        self._move_to_default_corner()

    def _status_dot_color(self) -> QtGui.QColor:
        if self._analysing:
            return QtGui.QColor("#FFD740")
        if is_monitoring_paused():
            return QtGui.QColor("#757575")
        if self._in_llm:
            return QtGui.QColor("#00E676")
        return QtGui.QColor("#757575")

    def _tick_dot_pulse(self) -> None:
        if not self._analysing:
            self._dot_pulse_timer.stop()
            self._dot_opacity = 1.0
            self.update()
            return
        self._dot_opacity = 0.4 if self._dot_opacity > 0.7 else 1.0
        self.update()

    def _start_dot_pulse(self) -> None:
        self._dot_opacity = 1.0
        self._dot_pulse_timer.start()

    def _stop_dot_pulse(self) -> None:
        self._dot_pulse_timer.stop()
        self._dot_opacity = 1.0

    def _toolbar_end_pos(self) -> QtCore.QPoint:
        """Dock toolbar to the right of the pill row in global screen coordinates."""
        if self._toolbar is None:
            return QtCore.QPoint(0, 0)
        tb_w = self._toolbar.width()
        tb_h = self._toolbar.sizeHint().height()
        base_w = self._content_width_for_layout()
        local_x = base_w - PILL_MARGIN_RIGHT + TOOLBAR_DOCK_MARGIN
        row_top = self._pill_row_top()
        # Align the toolbar to the pill row top so it reads as a slide-out side bar.
        local_y = row_top
        tb_top_left = self.mapToGlobal(QtCore.QPoint(local_x, local_y))
        end_global = QtCore.QPoint(tb_top_left.x() + tb_w, tb_top_left.y() + tb_h // 2)
        app = QtWidgets.QApplication.instance()
        scr = QtGui.QGuiApplication.screenAt(end_global) if app else None
        if scr is None and app:
            scr = app.primaryScreen()
        g = scr.availableGeometry() if scr else QtCore.QRect(0, 0, 1920, 1080)
        if end_global.x() > g.right() - 12:
            overflow = end_global.x() - (g.right() - 12)
            tb_top_left.setX(max(g.left() + 4, tb_top_left.x() - overflow))
        if tb_top_left.y() < g.top() + 4:
            tb_top_left.setY(g.top() + 4)
        if tb_top_left.y() + tb_h > g.bottom() - 4:
            tb_top_left.setY(max(g.top() + 4, g.bottom() - tb_h - 4))
        return tb_top_left

    def _toolbar_slide_start(self, end: QtCore.QPoint) -> QtCore.QPoint:
        """Slide animation offset in the same local coordinate system as _toolbar_end_pos."""
        ex, ey = end.x(), end.y()
        center_global = self.mapToGlobal(QtCore.QPoint(self.width() // 2, self.height() // 2))
        mid_x = center_global.x()
        if ex > mid_x:
            sx = ex + 24
        else:
            sx = ex - 24
        # Keep same y while sliding so hidden state does not leave a detached glyph artifact.
        sy = ey
        return QtCore.QPoint(sx, sy)

    def _update_toolbar_position(self) -> None:
        if self._toolbar is None or not self._toolbar.isVisible():
            return
        self._toolbar.move(self._toolbar_end_pos())

    def _ensure_toolbar(self) -> BubbleToolbar:
        if self._toolbar is not None:
            return self._toolbar
        self._toolbar = BubbleToolbar(self)
        self._toolbar.rephrase_clicked.connect(self._on_toolbar_rephrase)
        self._toolbar.redact_clicked.connect(self._on_toolbar_redact)
        self._toolbar.encrypt_clicked.connect(self._on_toolbar_encrypt)
        self._toolbar.scan_file_clicked.connect(self._on_toolbar_scan_file)
        self._toolbar.settings_clicked.connect(self._on_toolbar_settings)
        self._toolbar.rephrase_clicked.connect(self._hide_toolbar_immediately)
        self._toolbar.redact_clicked.connect(self._hide_toolbar_immediately)
        self._toolbar.encrypt_clicked.connect(self._hide_toolbar_immediately)
        self._toolbar.scan_file_clicked.connect(self._hide_toolbar_immediately)
        self._toolbar.settings_clicked.connect(self._hide_toolbar_immediately)
        return self._toolbar

    def _restart_toolbar_idle_timer(self) -> None:
        self._toolbar_idle_timer.stop()
        self._toolbar_idle_timer.start(4000)

    def _toolbar_content_enter(self) -> None:
        self._toolbar_hide_timer.stop()
        self._restart_toolbar_idle_timer()

    def _on_application_state_changed(self, state: QtCore.Qt.ApplicationState) -> None:
        if ALWAYS_SHOW_TOOLBAR:
            return
        if state != QtCore.Qt.ApplicationState.ApplicationActive:
            self._hide_toolbar_immediately()

    def _hide_toolbar(self) -> None:
        self._hide_toolbar_slide_out()

    def _should_keep_toolbar_visible(self) -> bool:
        if self.underMouse():
            return True
        if self._toolbar is not None:
            if self._toolbar.underMouse():
                return True
            for btn in self._toolbar.findChildren(QtWidgets.QWidget):
                if btn.underMouse():
                    return True
        return False

    def _maybe_hide_toolbar(self) -> None:
        if ALWAYS_SHOW_TOOLBAR:
            return
        if self._should_keep_toolbar_visible():
            return
        self._hide_toolbar()
        if self._tb_tooltip is not None:
            self._tb_tooltip.hide()

    def _hide_toolbar_immediately(self) -> None:
        if ALWAYS_SHOW_TOOLBAR:
            return
        self._hide_toolbar_for_drag()
        if self._tb_tooltip is not None:
            self._tb_tooltip.hide()

    def _hide_toolbar_idle_timeout(self) -> None:
        if ALWAYS_SHOW_TOOLBAR:
            return
        self._hide_toolbar_slide_out()

    def _stop_toolbar_anim(self) -> None:
        if self._toolbar_anim is not None:
            self._toolbar_anim.stop()
            self._toolbar_anim.deleteLater()
            self._toolbar_anim = None

    def _hide_toolbar_slide_out(self) -> None:
        if self._toolbar is None or not self._toolbar.isVisible():
            return
        tb = self._toolbar
        self._stop_toolbar_anim()
        end = self._toolbar_end_pos()
        slide_to = self._toolbar_slide_start(end)
        self._toolbar_anim = QtCore.QPropertyAnimation(tb, b"pos", self)
        self._toolbar_anim.setDuration(180)
        self._toolbar_anim.setStartValue(tb.pos())
        self._toolbar_anim.setEndValue(slide_to)
        self._toolbar_anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        self._toolbar_anim.finished.connect(self._on_toolbar_hide_anim_finished)
        self._toolbar_anim.start()

    def _on_toolbar_hide_anim_finished(self) -> None:
        if self._toolbar_anim is not None:
            try:
                self._toolbar_anim.finished.disconnect(self._on_toolbar_hide_anim_finished)
            except TypeError:
                pass
        self._stop_toolbar_anim()
        if self._toolbar is not None:
            self._toolbar.hide()
            # Reset dock position so Windows DWM does not leave a faded “ghost” at the slide-out spot.
            self._toolbar.move(self._toolbar_end_pos())
            self._toolbar.setWindowOpacity(1.0)
            self._refresh_window_geometry()

    def _hide_toolbar_for_drag(self) -> None:
        if self._toolbar is None or not self._toolbar.isVisible():
            return
        self._toolbar_hide_timer.stop()
        self._toolbar_idle_timer.stop()
        self._stop_toolbar_anim()
        self._toolbar.hide()
        self._refresh_window_geometry()

    def _show_toolbar(self) -> None:
        self._restart_toolbar_idle_timer()
        tb = self._ensure_toolbar()
        tb.adjustSize()
        self._refresh_window_geometry()
        # Keep the dock slot on-screen so the vertical menu is always visible.
        self.move(self._clamp_point_to_screen(self.pos()))
        if tb.isVisible():
            self._update_toolbar_position()
            return
        end = self._toolbar_end_pos()
        start = self._toolbar_slide_start(end)
        self._stop_toolbar_anim()
        tb.move(start)
        tb.show()
        tb.raise_()
        tb.setWindowOpacity(self.windowOpacity())
        self._toolbar_anim = QtCore.QPropertyAnimation(tb, b"pos", self)
        self._toolbar_anim.setDuration(180)
        self._toolbar_anim.setStartValue(start)
        self._toolbar_anim.setEndValue(end)
        self._toolbar_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._toolbar_anim.start()

    def _toggle_monitoring(self) -> None:
        self._on_toolbar_power()

    def _do_rephrase(self) -> None:
        self._on_toolbar_rephrase()

    def _do_redact(self) -> None:
        self._on_toolbar_redact()

    def _open_panel(self) -> None:
        self._on_delayed_single_click()

    def _do_encrypt(self) -> None:
        self._on_toolbar_encrypt()

    def _do_scan_file(self) -> None:
        import threading

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select document to scan for PII",
            "",
            "Documents (*.pdf *.docx *.xlsx *.pptx *.txt *.png *.jpg *.jpeg *.csv *.md)",
        )
        if not file_path:
            return

        show_toast(
            f"Scanning {Path(file_path).name}…",
            color="#1565C0",
            duration=30000,
            parent=self,
        )

        def _run() -> None:
            try:
                from document_scanner import scan_document

                def _progress(pct: int, msg: str) -> None:
                    self.document_scan_progress.emit(int(pct), str(msg))

                result = scan_document(
                    file_path,
                    use_gemini=True,
                    progress_cb=_progress,
                )
                self.document_scan_finished.emit(result)
            except Exception as e:
                self.document_scan_failed.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    @QtCore.pyqtSlot(object)
    def _on_scan_done(self, result: object) -> None:
        self.setToolTip("")
        if isinstance(result, dict) and result.get("error"):
            show_toast(
                f"Scan failed: {result['error']}",
                color="#C62828",
                parent=self,
            )
            return
        if not isinstance(result, dict):
            return
        count = int(result.get("issue_count", 0) or 0)
        risk = str(result.get("overall_risk", "low") or "low")
        color = {"high": "#C62828", "med": "#E65100", "low": "#2E7D32"}.get(risk, "#666")
        show_toast(
            f"Scan complete: {count} issues found — {risk.upper()} risk",
            color=color,
            parent=self,
        )
        report = result.get("report_path")
        if report:
            import webbrowser

            rp = Path(str(report))
            if rp.is_file():
                webbrowser.open(rp.as_uri())

    @QtCore.pyqtSlot(str)
    def _on_scan_failed(self, err: str) -> None:
        self.setToolTip("")
        show_toast(f"Scan error: {err}", color="#C62828", parent=self)

    def _on_scan_progress(self, pct: int, msg: str) -> None:
        self.setToolTip(f"Scan: {pct}% — {msg}")

    def _do_open_settings(self) -> None:
        self._on_toolbar_settings()

    def _on_toolbar_power(self) -> None:
        self._toggle_monitoring_power()
        self._apply_opacity_target()
        self.update()
        self._restart_toolbar_idle_timer()

    def _on_toolbar_rephrase(self) -> None:
        self._restart_toolbar_idle_timer()
        cb = QtGui.QGuiApplication.clipboard()
        text = (cb.text() or "").strip() or (self._last_text or "").strip()
        if not text:
            show_toast("No clipboard text to rephrase.", color="#757575", parent=self)
            return
        try:
            result = score_clipboard_with_pii(text)
            msg = str(result.get("message") or "").strip()
            if not msg:
                sug = result.get("suggestions")
                if isinstance(sug, list) and sug:
                    msg = str(sug[0])
            if not msg:
                msg = "Analysis complete."
            if len(msg) > 500:
                msg = msg[:500] + "…"
            show_toast(msg, color="#02C39A", duration=3500, parent=self)
        except Exception as e:
            show_toast(f"Rephrase failed: {e}", color="#E53935", parent=self)

    def _on_toolbar_redact(self) -> None:
        self._restart_toolbar_idle_timer()
        cb = QtGui.QGuiApplication.clipboard()
        text = (cb.text() or "").strip() or (self._last_text or "").strip()
        if not text:
            show_toast("No clipboard text.", color="#757575", parent=self)
            return
        spans = self._last_result.get("spans") if isinstance(self._last_result, dict) else []
        if isinstance(spans, list) and spans:
            out = mask_pii_spans(text, spans)
        else:
            out = mask_pii(text)
        cb.setText(out)
        show_toast("Masked and copied", color="#FFD740", parent=self)

    def _on_toolbar_encrypt(self) -> None:
        self._restart_toolbar_idle_timer()
        cb = QtGui.QGuiApplication.clipboard()
        text = (cb.text() or "").strip() or (self._last_text or "").strip()
        if not text:
            show_toast("No clipboard text.", color="#757575", parent=self)
            return
        try:
            enc = encrypt_text(text)
            cb.setText(enc)
            show_toast("Encrypted and copied", color="#42A5F5", parent=self)
        except Exception as e:
            show_toast(f"Encrypt failed: {e}", color="#E53935", parent=self)

    def _on_toolbar_scan_file(self) -> None:
        self._restart_toolbar_idle_timer()
        self._do_scan_file()

    def _on_toolbar_settings(self) -> None:
        self._restart_toolbar_idle_timer()
        self.context_menu_action.emit("settings")

    def _pill_target_color(self) -> QtGui.QColor:
        if not self._has_risk_score:
            return QtGui.QColor(0, 0, 0, 0)
        s = max(0, min(100, int(self._risk_score)))
        if s <= 40:
            return QtGui.QColor(27, 94, 32, 210)
        if s <= 70:
            return QtGui.QColor(230, 81, 0, 210)
        return QtGui.QColor(183, 28, 28, 210)

    def _sync_pill_bg(self) -> None:
        self._current_bg = self._pill_target_color()
        self.update()
        self._traffic_indicator.update()

    def set_hold_state(self, locked: bool, critical: bool = False) -> None:
        """Paste-hold: red pill, pulsing score ring, lock glyph until user remediates."""
        self._hold_critical = bool(critical)
        self._hold_locked = bool(locked)
        if locked:
            self._traffic_indicator.set_state("high")
            self._hold_pulse = True
            if self._hold_timer is None:
                self._hold_timer = QtCore.QTimer(self)
                self._hold_timer.setInterval(500)
                self._hold_timer.timeout.connect(self._pulse_hold)
            self._hold_timer.start()
        else:
            if self._hold_timer is not None:
                self._hold_timer.stop()
        self.update()
        self._traffic_indicator.update()

    def _pulse_hold(self) -> None:
        self._hold_pulse = not self._hold_pulse
        self.update()

    def _pill_fill_paint_color(self) -> QtGui.QColor:
        if getattr(self, "_hold_locked", False):
            return QtGui.QColor(183, 28, 28, 180)
        if is_monitoring_paused():
            return COLOR_PILL_PAUSED
        base = None
        if self._is_hovered and not self._has_risk_score:
            base = QtGui.QColor(0, 0, 0, 0)
        else:
            base = self._current_bg
        if (
            getattr(self, "_clipboard_preview_active", False)
            and not self._analysing
            and self._preview_bg.alpha() > 0
        ):
            tint = self._preview_bg
            a = tint.alpha() / 255.0
            return QtGui.QColor(
                int(base.red() * (1 - a) + tint.red() * a),
                int(base.green() * (1 - a) + tint.green() * a),
                int(base.blue() * (1 - a) + tint.blue() * a),
                min(255, int(base.alpha() * (1 - a) + tint.alpha() * a)),
            )
        return base

    def _shield_fill_color(self) -> QtGui.QColor:
        if is_monitoring_paused():
            return QtGui.QColor(0x7A, 0x7C, 0x7E)
        if not self._has_risk_score:
            return QtGui.QColor(0x96, 0x98, 0x9A)
        s = max(0, min(100, int(self._risk_score)))
        if s <= 40:
            return QtGui.QColor(0x69, 0xF0, 0xAE)
        if s <= 70:
            return QtGui.QColor(0xFF, 0xD5, 0x4F)
        return QtGui.QColor(0xEF, 0x9A, 0x9A)

    def _sync_toolbar_opacity(self) -> None:
        tb = self._toolbar
        if tb is not None and tb.isVisible():
            tb.setWindowOpacity(self.windowOpacity())
        self._traffic_indicator.setWindowOpacity(self.windowOpacity())

    def _sync_traffic_indicator_opacity(self) -> None:
        self._traffic_indicator.setWindowOpacity(self.windowOpacity())

    def _ambient_opacity(self) -> float:
        """Idle opacity (higher on hover). Kept readable on bright / HDR displays."""
        if self._critical_task_count > 0:
            return 0.55
        return 0.72

    def _tick_opacity(self) -> None:
        target = self._target_opacity
        current = self.windowOpacity()
        diff = target - current
        if abs(diff) < 0.02:
            self.setWindowOpacity(target)
            self._sync_toolbar_opacity()
            self._opacity_timer.stop()
            return
        self.setWindowOpacity(current + diff * 0.25)
        self._sync_toolbar_opacity()

    def _apply_opacity_target(self, *, animate: bool = True) -> None:
        self._target_opacity = 1.0 if self._is_hovered else self._ambient_opacity()
        if animate:
            self._opacity_timer.start()
        else:
            self.setWindowOpacity(self._target_opacity)
            self._sync_toolbar_opacity()
            self._opacity_timer.stop()

    def _tooltip_status_text(self) -> str:
        if is_monitoring_paused():
            base = "Monitoring paused"
        elif not self._in_llm:
            base = "Not monitoring"
        elif self._critical_task_count > 0:
            c = self._critical_task_count
            base = f"{c} critical item{'s' if c != 1 else ''} — review clipboard"
        elif self._issue_count > 0:
            n = self._issue_count
            base = f"{n} issue{'s' if n != 1 else ''} found — click to review"
        else:
            name = get_active_llm_name() or self._llm_target or "LLM"
            base = f"Monitoring: {name}"
        count = int(self._streak.get("streak_count", 0) or 0)
        if count >= 3:
            streak_text = f"🔥 {count} safe streak"
        elif count > 0:
            streak_text = f"✓ {count} safe"
        else:
            streak_text = "Start a streak!"
        return f"{base}\n{streak_text}"

    def _ensure_tooltip_win(self) -> QtWidgets.QLabel:
        if self._tooltip_win is None:
            lab = QtWidgets.QLabel(None)
            lab.setWindowFlags(
                QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.Tool
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
            )
            lab.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            lab.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            lab.setStyleSheet(
                "background: rgba(28,28,30,240); color: white; font-size: 11px; "
                "padding: 4px 8px; border-radius: 6px; border: none; font-family: 'Segoe UI';"
            )
            eff = QtWidgets.QGraphicsOpacityEffect(lab)
            lab.setGraphicsEffect(eff)
            eff.setOpacity(0.0)
            self._tooltip_win = lab
        return self._tooltip_win

    def _show_hover_tooltip(self) -> None:
        lab = self._ensure_tooltip_win()
        lab.setText(self._tooltip_status_text())
        lab.adjustSize()
        br = self.frameGeometry()
        x = br.left() + (self.width() - lab.width()) // 2
        y = br.bottom() + 4
        lab.move(x, y)
        eff = lab.graphicsEffect()
        if isinstance(eff, QtWidgets.QGraphicsOpacityEffect):
            eff.setOpacity(1.0)
        lab.setVisible(True)
        lab.raise_()
        self._hover_tooltip_visible = True
        if self._status_hide_timer is None:
            self._status_hide_timer = QtCore.QTimer(self)
            self._status_hide_timer.setSingleShot(True)
            self._status_hide_timer.timeout.connect(self._hide_hover_tooltip)
        self._status_hide_timer.stop()
        self._status_hide_timer.start(1500)

    def _hide_hover_tooltip(self) -> None:
        if self._status_hide_timer is not None:
            self._status_hide_timer.stop()
        if self._tooltip_win is None:
            self._hover_tooltip_visible = False
            return
        self._tooltip_win.setVisible(False)
        self._hover_tooltip_visible = False

    def _on_hover_enter_deferred(self) -> None:
        self._show_hover_tooltip()

    def _on_hover_leave_deferred(self) -> None:
        self._hide_hover_tooltip()
        w = QtWidgets.QApplication.widgetAt(QtGui.QCursor.pos())
        p: Optional[QtWidgets.QWidget] = w
        while p is not None:
            if p is self or p is self._toolbar:
                return
            p = p.parentWidget()
        self._target_opacity = self._ambient_opacity()
        self._is_hovered = False
        self._opacity_timer.start()
        self.update()
        self._toolbar_hide_timer.stop()
        self._toolbar_hide_timer.start(400)

    def enterEvent(self, event: QtCore.QEvent) -> None:
        st = self._state.lower()
        if st in ("idle", "none", "not_monitoring"):
            self._target_width = COLLAPSED_W
        else:
            count = int(self._streak.get("streak_count", 0) or 0)
            self._target_width = EXPANDED_W_NO_STREAK if count == 0 else EXPANDED_W
        self._expand_timer.start()
        self._toolbar_hide_timer.stop()
        if self._status_hide_timer is not None:
            self._status_hide_timer.stop()
        self._target_opacity = 1.0
        self._is_hovered = True
        self._opacity_timer.start()
        self._show_toolbar()
        self.update()
        self._traffic_indicator.update()
        QtCore.QTimer.singleShot(0, self._on_hover_enter_deferred)
        QtCore.QTimer.singleShot(0, self._maybe_show_overlay_if_hovered)
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._target_width = COLLAPSED_W
        self._expand_timer.start()
        QtCore.QTimer.singleShot(0, self._on_hover_leave_deferred)
        self._traffic_indicator.update()
        super().leaveEvent(event)

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:
        super().moveEvent(event)
        if self._toolbar is not None and self._toolbar.isVisible():
            self._update_toolbar_position()
        self._reposition_traffic_light()

    def bring_to_front(self) -> None:
        """Raise pill + traffic-light chrome (Windows multi-monitor / Z-order)."""
        self.raise_()
        if self._traffic_indicator is not None:
            self._traffic_indicator.raise_()
        if self._toolbar is not None and self._toolbar.isVisible():
            self._toolbar.raise_()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._traffic_indicator.show()
        self._reposition_traffic_light()
        self._sync_traffic_indicator_opacity()
        if ALWAYS_SHOW_TOOLBAR:
            QtCore.QTimer.singleShot(0, self._show_toolbar)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self._traffic_indicator.hide()
        if self._toolbar is not None:
            self._toolbar.hide()
        super().hideEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        pw = self._pill_width_f()
        body = _pill_body_path(pw)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(self._pill_fill_paint_color())
        p.drawPath(body)
        # No outer border stroke on the pill (fill only — avoids faint rectangular outline).
        # Bridge the segment join by 1px to avoid subpixel seams on some DPI scales.
        p.fillRect(
            QtCore.QRectF(float(PILL_X) - 0.5, float(PILL_Y), 1.0, float(PILL_H)),
            self._pill_fill_paint_color(),
        )
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtCore.Qt.PenStyle.NoPen)

        p.save()
        sc_badge = int(self._streak.get("streak_count", 0) or 0)
        if sc_badge >= 10:
            p.setPen(QtGui.QColor("#FFB300"))
            fire_font = QtGui.QFont()
            fire_font.setPixelSize(8)
            p.setFont(fire_font)
            p.drawText(
                QtCore.QRectF(float(PILL_X) + 4, float(PILL_Y) - 8, 20.0, 10.0),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"🔥{sc_badge}",
            )
        p.restore()

        # Status indicator above power glyph (requested: 4x4).
        dot_r = QtCore.QRectF(float(PILL_X + 5), float(PILL_Y + 5), 4.0, 4.0)
        dc = self._status_dot_color()
        p.save()
        if self._analysing:
            p.setOpacity(self._dot_opacity)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(dc)
        p.drawEllipse(dot_r)
        p.restore()

        pcol = QtGui.QColor(0x02, 0xC3, 0x9A)
        if is_monitoring_paused():
            pcol = QtGui.QColor(120, 120, 122)
        p.setPen(pcol)
        fam = p.font().family() or "Segoe UI"
        p.setFont(paint_font_px(fam, 12, floor=MIN_PX_BODY))
        p.drawText(
            QtCore.QRectF(float(PILL_X + 1), float(PILL_Y + 2), 12.0, float(PILL_H - 4)),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "⏻",
        )

        sy = PILL_Y + (PILL_H - 16) // 2
        sx = PILL_X + 14
        shield_fill = self._shield_fill_color()
        if self._current_width > 100:
            p.setPen(shield_fill)
            p.setFont(paint_font_px(fam, 14, floor=MIN_PX_LABEL))
            p.drawText(
                QtCore.QRectF(float(sx), float(sy), 16.0, 16.0),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "◈",
            )
            score_slot = QtCore.QRectF(float(sx + 16), float(PILL_Y + 2), 28.0, float(PILL_H - 4))
        else:
            score_slot = QtCore.QRectF(
                float(PILL_X + 14),
                float(PILL_Y + 2),
                float(pw) - 18.0,
                float(PILL_H - 4),
            )
        if getattr(self, "_hold_locked", False) and not self._analysing:
            lock_w = 13.0
            p.setPen(shield_fill)
            p.setFont(paint_font_px(fam, 10, floor=MIN_PX_LABEL))
            p.drawText(
                QtCore.QRectF(
                    float(score_slot.x()),
                    float(score_slot.y()),
                    lock_w,
                    float(score_slot.height()),
                ),
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                "🔒",
            )
            score_slot = QtCore.QRectF(
                float(score_slot.x() + lock_w),
                float(score_slot.y()),
                max(4.0, float(score_slot.width() - lock_w)),
                float(score_slot.height()),
            )
        if not self._analysing:
            if self._critical_task_count > 0:
                p.setPen(COLOR_CRITICAL_COUNT)
                p.setFont(paint_font_px(fam, 11, bold=True, floor=MIN_PX_LABEL))
                p.drawText(
                    score_slot,
                    QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter,
                    str(int(self._critical_task_count)),
                )
            elif self._has_risk_score and int(self._risk_score) >= 30:
                p.setPen(shield_fill)
                p.setFont(paint_font_px(fam, 11, bold=True, floor=MIN_PX_LABEL))
                p.drawText(
                    score_slot,
                    QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                    str(int(self._risk_score)),
                )
            else:
                pm = self._get_idle_pixmap()
                if pm is not None and not pm.isNull():
                    p.save()
                    p.setOpacity(0.65)
                    diameter = max(8.0, min(float(score_slot.width()), float(score_slot.height())) - 2.0)
                    badge_rect = QtCore.QRectF(
                        float(score_slot.x() + (score_slot.width() - diameter) / 2.0),
                        float(score_slot.y() + (score_slot.height() - diameter) / 2.0),
                        diameter,
                        diameter,
                    )
                    scaled = pm.scaled(
                        int(diameter),
                        int(diameter),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                    x = int(badge_rect.x() + (badge_rect.width() - scaled.width()) / 2.0)
                    y = int(badge_rect.y() + (badge_rect.height() - scaled.height()) / 2.0)
                    clip = QtGui.QPainterPath()
                    clip.addEllipse(badge_rect)
                    p.setClipPath(clip)
                    p.drawPixmap(x, y, scaled)
                    p.setClipping(False)
                    p.restore()
                else:
                    p.save()
                    diameter = max(9.0, min(float(score_slot.width()), float(score_slot.height())) - 1.0)
                    badge_rect = QtCore.QRectF(
                        float(score_slot.x() + (score_slot.width() - diameter) / 2.0),
                        float(score_slot.y() + (score_slot.height() - diameter) / 2.0),
                        diameter,
                        diameter,
                    )
                    p.setPen(QtCore.Qt.PenStyle.NoPen)
                    p.setBrush(QtGui.QColor(12, 37, 34, 220))
                    p.drawEllipse(badge_rect)
                    p.setPen(QtGui.QColor(255, 255, 255, 220))
                    p.setFont(paint_font_px(fam, 9, bold=True, floor=MIN_PX_LABEL))
                    p.drawText(
                        badge_rect,
                        QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter,
                        "CS",
                    )
                    p.restore()

        pcx = PILL_X + int(pw) // 2
        pcy = PILL_Y + PILL_H // 2

        if self._analysing and self._spinner_opacity > 0.01:
            p.save()
            p.setOpacity(self._spinner_opacity)
            pen = QtGui.QPen(COLOR_SPINNER, 2.0)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawArc(
                QtCore.QRectF(pcx - 12, pcy - 12, 24, 24),
                int(-self._spin_angle * 16),
                int(-270 * 16),
            )
            p.restore()

        if self._show_all_clear and not self._issue_count:
            p.setPen(COLOR_CHECK)
            p.setFont(paint_font_px(fam, 12, bold=True, floor=MIN_PX_BODY))
            p.drawText(
                QtCore.QRectF(float(pcx + 8), float(pcy - 8), 16.0, 16.0),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "✓",
            )

        self._paint_expanded_hover_content(p, fam)
        # Keep chrome minimal: do not draw floating score ring/capsule.

        # Intentionally hide the compact clipboard-preview badge to keep the overlay clean.

    def set_clipboard_preview(self, level: str, score: int) -> None:
        """
        Updates traffic light and pill tint from background clipboard scoring (not a paste event).
        """
        self._clipboard_preview_active = True
        self._preview_level = str(level or "safe").lower()
        try:
            self._preview_score = max(0, min(100, int(score)))
        except (TypeError, ValueError):
            self._preview_score = 0
        color_map = {
            "high": QtGui.QColor(229, 57, 53, 120),
            "med": QtGui.QColor(255, 179, 0, 100),
            "safe": QtGui.QColor(67, 160, 71, 80),
        }
        self._preview_bg = color_map.get(self._preview_level, QtGui.QColor(28, 28, 30, 0))
        state_map = {"high": "high", "med": "med", "safe": "safe"}
        if self._traffic_indicator is not None:
            self._traffic_indicator.set_state(state_map.get(self._preview_level, "safe"))
        self.update()

    def clear_clipboard_preview(self) -> None:
        """End preview tint / restore normal monitoring-driven traffic light."""
        self._clipboard_preview_active = False
        self._preview_level = "safe"
        self._preview_score = 0
        self._preview_bg = QtGui.QColor(28, 28, 30, 0)
        self._refresh_monitoring()

    def _reapply_clipboard_preview_tl(self) -> None:
        if not getattr(self, "_clipboard_preview_active", False):
            return
        if self._in_llm or is_monitoring_paused() or self._analysing:
            return
        sm = {"high": "high", "med": "med", "safe": "safe"}
        if self._traffic_indicator is not None:
            self._traffic_indicator.set_state(sm.get(self._preview_level, "safe"))

    def _refresh_monitoring(self) -> None:
        was = self._in_llm
        now = time.monotonic()
        # Match paste hook: same title keyword fallback; if "monitor all windows" is off in settings,
        # treat as in-context when not LLM-only so the pill does not stay stuck on "not monitoring".
        if get_monitor_llm_only():
            if self._analysing and (now - self._last_monitor_refresh_mono) < 2.5:
                self._in_llm = was
            else:
                self._in_llm = bool(detect_llm_window()[0])
        else:
            self._in_llm = True
        self._last_monitor_refresh_mono = now
        want_ms = 1500 if self._in_llm else 3500
        if self._monitor_timer.interval() != want_ms:
            self._monitor_timer.setInterval(want_ms)
        paused = is_monitoring_paused()
        target_state = "idle"
        if was != self._in_llm:
            if not self._in_llm:
                self.set_idle()
            elif not paused:
                self._traffic_indicator.set_state("idle")
        if not self._in_llm or paused:
            target_state = "not_monitoring"
            self._traffic_indicator.set_state(target_state)
            self._reapply_clipboard_preview_tl()
        elif not self._analysing:
            target_state = "idle"
        new_state = (self._in_llm, paused, target_state, bool(self._hover_tooltip_visible))
        if new_state != self._last_monitor_state:
            self._last_monitor_state = new_state
            self.update()
        if self._hover_tooltip_visible and self._tooltip_win is not None:
            self._tooltip_win.setText(self._tooltip_status_text())
            self._tooltip_win.adjustSize()
        if was != self._in_llm:
            self._apply_opacity_target()

    def set_analysing(self, text: str = "") -> None:
        self.clear_clipboard_preview()
        self._spinner_timer.stop()
        self._hide_feedback_buttons()
        if text:
            self._last_text = text
        self._last_label = ""
        self._all_clear_timer.stop()
        self._show_all_clear = False
        self._issue_count = 0
        self._critical_task_count = 0
        self._analysing = True
        self._spin_angle = 0.0
        self._spinner_opacity = 1.0
        self._start_dot_pulse()
        self._spinner_timer.start()
        self.update()
        self._traffic_indicator.set_state("analysing")
        self._apply_opacity_target()

    def set_idle(self) -> None:
        self._hide_feedback_buttons()
        self._spinner_timer.stop()
        self._stop_dot_pulse()
        self._has_risk_score = False
        self._sync_pill_bg()
        self._analysing = False
        self._spinner_opacity = 0.0
        self._show_all_clear = False
        self._issue_count = 0
        self._critical_task_count = 0
        self._all_clear_timer.stop()
        self._traffic_indicator.set_state("not_monitoring")
        self._reapply_clipboard_preview_tl()
        self.update()
        self._apply_opacity_target()

    def set_clear(self) -> None:
        self._spinner_timer.stop()
        self._stop_dot_pulse()
        self._analysing = False
        self._issue_count = 0
        self._critical_task_count = 0
        self._show_all_clear = True
        self._all_clear_timer.stop()
        self._spinner_opacity = 0.0
        self._all_clear_timer.start(3000)
        self._traffic_indicator.set_state("safe")
        self.update()
        self.show_feedback_buttons(self._last_text, self._last_label)
        self._apply_opacity_target()

    def set_issues(self, count: int) -> None:
        n = max(0, int(count))
        if n <= 0:
            self.set_clear()
            return

        self._spinner_timer.stop()
        self._stop_dot_pulse()
        self._analysing = False
        self._show_all_clear = False
        self._all_clear_timer.stop()
        self._issue_count = n
        self._spinner_opacity = 0.0
        rl = str(self._last_label or "low").lower()
        action = str(self._result.get("action", "") or "")
        if rl in ("high", "block") or action == "block":
            self._traffic_indicator.set_state("high")
        elif rl == "med":
            self._traffic_indicator.set_state("med")
        else:
            self._traffic_indicator.set_state("safe")
        self.update()
        self.show_feedback_buttons(self._last_text, self._last_label)
        self._apply_opacity_target()

    def _tick_spinner(self) -> None:
        if not self._analysing:
            self._spinner_timer.stop()
            return
        self._spin_angle = (self._spin_angle + 36.0) % 360.0
        self.update()

    def _hide_all_clear_indicator(self) -> None:
        self._show_all_clear = False
        self.update()

    def clear_issue_badge(self) -> None:
        self._issue_count = 0
        self._critical_task_count = 0
        if not self._analysing:
            self._spinner_timer.stop()
        self._traffic_indicator.set_state("safe")
        self.update()
        self._apply_opacity_target()

    def set_show_badge_on_issues(self, show: bool) -> None:
        self._show_badge_on_issues = bool(show)
        self.update()

    def _open_user_dashboard(self) -> None:
        """Open hosted dashboard first; fall back to local generated dashboard."""
        hosted = (
            os.environ.get("GUARDRAIL_HOSTED_DASHBOARD_URL", "")
            or "https://alwinphilip.online/Core_Sentinal/"
        ).strip()
        if hosted:
            try:
                opened = bool(webbrowser.open(hosted))
                if opened:
                    return
            except Exception as e:
                print(f"[dashboard] hosted open failed: {e}")
            # Windows fallback if webbrowser handler fails/returns False.
            try:
                os.startfile(hosted)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        root = Path(__file__).resolve().parent
        gen_path = root / "reports" / "user_dashboard_gen.py"
        if not gen_path.exists():
            show_toast(
                "Hosted dashboard unavailable and local dashboard generator not found.",
                color="#E65100",
                parent=self,
            )
            return
        try:
            spec = importlib.util.spec_from_file_location(
                "user_dashboard_gen",
                gen_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("user_dashboard_gen spec missing")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            path = mod.generate(
                db_path=str(root / "logs" / "guardrail.db"),
                settings_path=str(root / "config" / "user_settings.json"),
                output_path=str(root / "logs" / "user_dashboard.html"),
            )
            os.startfile(os.path.abspath(path))
        except Exception as e:
            print(f"[dashboard] {e}")
            p = root / "logs" / "user_dashboard.html"
            if p.exists():
                os.startfile(str(p.resolve()))

    def _show_context_menu(self, global_pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #2b2b2b; color: #e8e8e8; border: 1px solid #444; "
            "padding: 4px; font-family: 'Segoe UI'; font-size: 12px; }"
            "QMenu::item { padding: 6px 24px 6px 12px; }"
            "QMenu::item:selected { background: #3d3d3d; }"
            "QMenu::separator { height: 1px; background: #444; margin: 4px 8px; }"
        )

        act_settings = menu.addAction("Settings")
        act_settings.triggered.connect(lambda: self.context_menu_action.emit("settings"))

        act_pause = menu.addAction("Pause monitoring")
        act_pause.setCheckable(True)
        act_pause.setChecked(is_monitoring_paused())
        act_pause.triggered.connect(self.monitoring_pause_changed.emit)

        menu.addSeparator()
        act_scan = menu.addAction("Scan file for PII...")
        act_scan.triggered.connect(lambda: self.context_menu_action.emit("scan_file"))
        act_report = menu.addAction("View report")
        act_report.triggered.connect(lambda: self.context_menu_action.emit("view_report"))
        menu.addSeparator()
        quit_action = QtGui.QAction("Quit", menu)
        quit_action.triggered.connect(lambda: self.context_menu_action.emit("quit"))
        menu.addAction(quit_action)
        dash_action = QtGui.QAction("📊 My Privacy Dashboard", menu)
        dash_action.triggered.connect(self._open_user_dashboard)
        menu.insertAction(quit_action, dash_action)
        menu.insertSeparator(quit_action)

        menu.exec(global_pos)

    def update_from_result(
        self,
        result: dict,
        llm_target: str,
        original_text: str = "",
        url: Optional[str] = None,
        cleaned_title: Optional[str] = None,
    ) -> None:
        self.clear_clipboard_preview()
        r: dict = result if isinstance(result, dict) else {}
        self._result = r
        self._last_result = dict(r)
        self._llm_target = llm_target or ""
        self._original_text = original_text or ""
        self._url = url or ""
        self._cleaned_title = cleaned_title or ""

        action = r.get("action", "")
        if action not in ("silent", "warn", "block"):
            d = r.get("decision", "allow")
            action = {"allow": "silent", "warn": "warn", "block": "block"}.get(d, "silent")
        triggers = r.get("triggers") or []
        if not isinstance(triggers, (list, tuple)):
            triggers = []
        spans = r.get("spans") if isinstance(r.get("spans"), list) else []
        try:
            score = int(r.get("risk_score", 0))
        except (TypeError, ValueError):
            score = 0
        critical = bool(r.get("critical_secret_detected", False))

        risk = r.get("risk", "low")
        self._last_text = (original_text or "") or self._last_text
        self._last_label = str(risk) if risk else "low"

        self._llm_name = (
            str(r.get("llm_name") or "").strip()
            or str(get_active_llm_name() or "").strip()
            or (self._llm_name or "")
            or "Not monitoring"
        )
        self._last_action = str(action)

        clear_path = action == "silent" and not critical and score < 30 and not triggers
        span_len = len(spans)
        if clear_path:
            self.set_clear()
        else:
            n = len(triggers)
            if n <= 0 and (action in ("warn", "block") or critical or score >= 30):
                n = 1
            # Badge count must not drop to 0 when the model omitted `spans` but flagged risk via triggers/action.
            self.set_issues(max(span_len, n))
            self._critical_task_count = _derive_critical_task_count(r)

        self._risk_score = int(r.get("risk_score", 0) or 0)
        self._last_score = self._risk_score
        self._has_risk_score = True
        self._sync_pill_bg()

        self._target_score = self._risk_score
        if abs(self._displayed_score - float(self._target_score)) > 0.01:
            self._ring_timer.start()

        self.update()
        self.show_character(str(risk) if risk is not None else "low", score, spans)

    def _ensure_character_widget(self) -> CharacterWidget:
        if self._character_widget is None:
            self._character_widget = CharacterWidget()
        return self._character_widget

    def show_character(self, risk_label: str, score: int, spans: list) -> None:
        """
        Character reaction after each scored paste. Does not alter risk / pill state.
        """
        del spans
        r = self._result if isinstance(self._result, dict) else {}
        risk = str(risk_label or "low").lower()
        critical = bool(r.get("critical_secret_detected", False))
        try:
            sc = int(r.get("risk_score", 0) or 0)
        except (TypeError, ValueError):
            sc = int(score) if isinstance(score, int) else 0
        triggers = r.get("triggers") or []
        if not isinstance(triggers, (list, tuple)):
            triggers = []
        action = r.get("action", "")
        if action not in ("silent", "warn", "block"):
            d = r.get("decision", "allow")
            action = {"allow": "silent", "warn": "warn", "block": "block"}.get(d, "silent")
        clear_path = action == "silent" and not critical and sc < 30 and not triggers

        # Silent score 0–40: main shows CharacterEvent near the pill; skip CharacterWidget here.
        if action == "silent" and not critical and sc <= 40 and not triggers:
            return

        if not self._character_first_scored_done:
            self._character_first_scored_done = True
            if clear_path and risk == "low":
                ch = self._ensure_character_widget()
                ch.configure(
                    "😌",
                    "All clear! Safe to paste.",
                    "#2E7D32",
                )
                ch.show_near_traffic_light(self)
                return

        if clear_path and risk == "low":
            return

        if critical:
            ch = self._ensure_character_widget()
            ch.configure(
                "🤦",
                "This would expose your credentials!",
                "#C62828",
                large_text=True,
            )
            ch.show_near_traffic_light(self)
            return

        if risk == "high":
            n = user_settings.record_high_risk_paste()
            if n == 1:
                msg = "Stop! Sensitive data detected."
            elif n == 2:
                msg = "Again? Please review before pasting."
            else:
                msg = f"That's {n} times today — be careful!"
            badge = f"x{n} today" if n >= 2 else None
            ch = self._ensure_character_widget()
            ch.configure(
                "😱",
                msg,
                "#C62828",
                streak_badge=badge,
            )
            ch.show_near_traffic_light(self)
            return

        if risk == "med" or (not clear_path and risk == "low"):
            ch = self._ensure_character_widget()
            ch.configure(
                "😬",
                "Heads up — check this before pasting.",
                "#E65100",
            )
            ch.show_near_traffic_light(self)
            return

    def show_feedback_buttons(self, original_text: str, predicted_label: str) -> None:
        if original_text:
            self._last_text = original_text
        if predicted_label:
            self._last_label = predicted_label
        if not (self._last_text or self._original_text or "").strip():
            return
        if self._feedback_survey_timer is None:
            self._feedback_survey_timer = QtCore.QTimer(self)
            self._feedback_survey_timer.setSingleShot(True)
            self._feedback_survey_timer.timeout.connect(self._end_feedback_survey)
        self._feedback_survey_timer.stop()
        self._dismiss_overlay_feedback_prompt()
        self._feedback_survey_active = True
        self._feedback_survey_timer.start(8000)
        QtCore.QTimer.singleShot(0, self._maybe_show_overlay_if_hovered)

    def _maybe_show_overlay_if_hovered(self) -> None:
        """During the 8s post-detection window, show the prompt when the pill is hovered."""
        if not self._feedback_survey_active:
            return
        if not self._is_hovered:
            return
        if self._overlay_feedback_prompt is not None:
            return
        self._show_overlay_feedback_prompt()

    def _end_feedback_survey(self) -> None:
        self._feedback_survey_active = False
        self._dismiss_overlay_feedback_prompt()

    def _stop_feedback_survey(self) -> None:
        self._feedback_survey_active = False
        if self._feedback_survey_timer is not None:
            self._feedback_survey_timer.stop()

    def _show_overlay_feedback_prompt(self) -> None:
        """Compact Correct/Wrong prompt below the pill (shown on hover during post-detection window)."""
        if not (self._last_text or self._original_text or "").strip():
            return
        self._dismiss_overlay_feedback_prompt()
        w = QtWidgets.QWidget(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool,
        )
        w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        w.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        lbl = QtWidgets.QLabel("Correct?")
        lbl.setStyleSheet(
            "color: rgba(255,255,255,0.6);"
            "font-size: 10px; background: transparent;"
        )
        layout.addWidget(lbl)

        for txt, color, action in (
            ("✓", "#43A047", "correct"),
            ("✗", "#E53935", "wrong"),
        ):
            btn = QtWidgets.QPushButton(txt)
            btn.setFixedSize(24, 24)
            btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background: {color}33;
                    color: {color};
                    border: 1px solid {color}66;
                    border-radius: 5px;
                    font-size: 12px;
                    font-weight: 700;
                }}
                QPushButton:hover {{
                    background: {color}66;
                }}
                """
            )
            btn.clicked.connect(
                lambda _c=False, a=action, ww=w: self._handle_overlay_feedback(a, ww)
            )
            layout.addWidget(btn)

        w.setStyleSheet(
            "QWidget { background: rgba(22,22,24,235); border-radius: 8px; "
            "border: none; }"
        )
        w.adjustSize()
        pill_global = self.mapToGlobal(QtCore.QPoint(0, 0))
        w.move(pill_global.x(), pill_global.y() + self.height() + 6)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            ag = screen.availableGeometry()
            x = w.x()
            y = w.y()
            x = max(ag.left() + 4, min(x, ag.right() - w.width() - 4))
            y = min(y, ag.bottom() - w.height() - 8)
            w.move(x, y)
        w.show()
        w.raise_()
        self._overlay_feedback_prompt = w
        self._feedback_prompt = w

    def _dismiss_overlay_feedback_prompt(self) -> None:
        ow = self._overlay_feedback_prompt
        if ow is not None:
            ow.hide()
            ow.deleteLater()
            self._overlay_feedback_prompt = None
            self._feedback_prompt = None

    def _handle_overlay_feedback(self, action: str, widget: QtWidgets.QWidget) -> None:
        self._stop_feedback_survey()
        widget.hide()
        widget.deleteLater()
        if self._overlay_feedback_prompt is widget:
            self._overlay_feedback_prompt = None
            self._feedback_prompt = None
        if action == "correct":
            self._record_overlay_feedback("correct")
            from toast import toast_safe

            toast_safe("Confirmed — model noted!")
        else:
            self._record_overlay_feedback("wrong")
            cb = self._open_panel_callback
            if callable(cb):
                cb()
            else:
                self._on_delayed_single_click()

    def _record_overlay_feedback(self, kind: str) -> None:
        from feedback_store import record_feedback

        text = (self._last_text or self._original_text or "").strip()
        if not text:
            return
        pred = str(self._last_label or "low").lower()
        try:
            rs = int(self._result.get("risk_score", 0) or 0)
        except (TypeError, ValueError):
            rs = 0
        if kind == "correct":
            record_feedback(
                text,
                pred,
                pred,
                source="overlay_correct",
                risk_score=rs,
                feedback_type="correct",
            )
        elif kind == "wrong":
            record_feedback(
                text,
                pred,
                pred,
                source="overlay_wrong",
                risk_score=rs,
                feedback_type="wrong_intent",
            )

    def _hide_feedback_buttons(self) -> None:
        if self._feedback_timer is not None:
            self._feedback_timer.stop()
        if self._feedback_widget is not None:
            self._feedback_widget.hide()
        self._stop_feedback_survey()
        self._dismiss_overlay_feedback_prompt()

    def show_near_bottom_right(self) -> None:
        self._move_to_default_corner()

    def show_near_cursor(self) -> None:
        pos = QtGui.QCursor.pos()
        x = pos.x() - self.width() - 8
        y = pos.y() - self.height() - 8
        screen = QtGui.QGuiApplication.screenAt(pos)
        if screen:
            g = screen.availableGeometry()
            x = max(g.left(), min(x, g.right() - self.width()))
            y = max(g.top(), min(y, g.bottom() - self.height()))
        self.move(x, y)

    def _on_indicator_remediation_click(self) -> None:
        self._on_delayed_single_click()

    def _on_delayed_single_click(self) -> None:
        r = self._result if isinstance(self._result, dict) else {}
        try:
            rs = int(r.get("risk_score", 0) or 0)
        except (TypeError, ValueError):
            rs = 0
        has_text = bool((self._original_text or self._last_text or "").strip())
        has_triggers_or_spans = bool(r.get("triggers")) or bool(r.get("spans"))
        if (
            self._issue_count <= 0
            and rs <= 0
            and not has_text
            and not has_triggers_or_spans
            and str(r.get("action", "") or "") not in ("warn", "block")
        ):
            return
        from ui_remediation_dialog import RemediationDialog

        existing = self._remediation_panel
        if (
            existing is not None
            and existing.isVisible()
            and not getattr(existing, "_closing_anim", False)
        ):
            existing.slide_out_and_hide()
            return

        if existing is not None:
            same = (
                getattr(existing, "_original_text", "") == (self._original_text or "")
                and getattr(existing, "_result", {}) == self._result
                and getattr(existing, "_llm_target", "") == (self._llm_target or "")
            )
            if not same:
                existing.deleteLater()
                self._remediation_panel = None

        self.hide()
        if self._remediation_panel is None:
            dialog = RemediationDialog(
                self._original_text,
                self._result,
                self._llm_target,
                self,
            )
            self._remediation_panel = dialog
            app = QtWidgets.QApplication.instance()
            pc = app.property("paste_controller") if app else None
            if pc is not None:
                dialog.monitoring_toggled.connect(
                    lambda paused: pc.kick_queue() if not paused else None
                )

            def _after_remediation(code: int) -> None:
                self.show()
                if code == QtWidgets.QDialog.DialogCode.Accepted:
                    self.clear_issue_badge()

            def _after_slide_out(_ok: bool) -> None:
                self.show()

            dialog.finished.connect(_after_remediation)
            dialog.remediation_finished.connect(_after_slide_out)

        panel = self._remediation_panel
        assert panel is not None
        panel.show()

    def _power_hit_rect(self) -> QtCore.QRect:
        return QtCore.QRect(PILL_X + 1, PILL_Y + 2, 12, PILL_H - 4)

    def _toggle_monitoring_power(self) -> None:
        paused = not is_monitoring_paused()
        set_monitoring_paused(paused)
        self.monitoring_pause_changed.emit(paused)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            if self._power_hit_rect().contains(pos):
                self._toggle_monitoring_power()
                self._apply_opacity_target()
                self.update()
                event.accept()
                return
            self._dragging = True
            self._hide_toolbar_immediately()
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            self._press_pos = event.globalPosition().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._hide_toolbar_for_drag()
            new_pos = event.globalPosition().toPoint() - self._drag_offset
            new_pos = self._clamp_point_to_screen(new_pos)
            self.move(new_pos)
            self._update_toolbar_position()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._click_timer.stop()
            self._dragging = False
            self._press_pos = None
            self.menu_action_clicked.emit("redact")
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            try:
                user_settings.save_bubble_position(self.x(), self.y())
            except Exception:
                pass
            moved = 0
            if self._press_pos is not None:
                moved = (event.globalPosition().toPoint() - self._press_pos).manhattanLength()
            self._press_pos = None
            if moved < 5:
                self._click_timer.start()
            self._update_toolbar_position()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _clear_action_card_layout(self) -> None:
        if self._action_card_reuse is None:
            return
        lay = self._action_card_reuse.layout()
        if lay is None:
            return
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _ensure_action_card_frame(self) -> QtWidgets.QFrame:
        if self._action_card_reuse is None:
            self._action_card_reuse = QtWidgets.QFrame()
            self._action_card_reuse.setObjectName("actionCard")
            self._action_card_reuse.setStyleSheet(CARD_QSS)
        self._clear_action_card_layout()
        return self._action_card_reuse

    def _dismiss_action_card(self) -> None:
        self._redact_toast_timer.stop()
        if self._action_card_wrap.height() <= 0:
            if self._card_anim:
                self._card_anim.stop()
                self._card_anim = None
            self._clear_action_card_layout()
            return
        pill_cy = self._pill_center_global().y()
        if self._card_anim:
            self._card_anim.stop()
            self._card_anim = None
        self._clear_action_card_layout()
        self._action_card_wrap.setFixedHeight(0)
        self._action_card_wrap.move(0, 0)
        self._resize_to_state()
        self.move(self.x(), int(pill_cy - self._pill_center_offset_from_window_top()))

    def _slide_in_card(self, frame: QtWidgets.QWidget) -> None:
        if self._card_anim:
            self._card_anim.stop()
        self._card_anim = QtCore.QPropertyAnimation(frame, b"pos", self._action_card_wrap)
        self._card_anim.setDuration(220)
        self._card_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._card_anim.setStartValue(frame.pos())
        self._card_anim.setEndValue(QtCore.QPoint(frame.x(), 4))
        self._card_anim.start()

    def _mount_action_card(self, frame: QtWidgets.QWidget) -> None:
        if self._card_anim:
            self._card_anim.stop()
            self._card_anim = None
        frame.setParent(self._action_card_wrap)
        frame.setObjectName("actionCard")
        if not frame.styleSheet():
            frame.setStyleSheet(CARD_QSS)
        fw = CARD_W - 8
        frame.setFixedWidth(fw)
        frame.adjustSize()
        inner_h = frame.height()
        wrap_h = inner_h + 8
        pill_cy = self._pill_center_global().y()
        self._action_card_wrap.setFixedSize(CARD_W, wrap_h)
        self._action_card_wrap.move(0, 0)
        self._resize_to_state()
        x_m = max(4, (self.width() - fw) // 2)
        frame.move(x_m, 4 + CARD_SLIDE_PX)
        self._action_card_inner = frame
        self.move(self.x(), int(pill_cy - self._pill_center_offset_from_window_top()))
        frame.show()
        self._slide_in_card(frame)

    def handle_menu_action(self, action: str) -> None:
        text = (self._original_text or "").strip()
        if not text:
            QtWidgets.QToolTip.showText(
                QtGui.QCursor.pos(),
                "Nothing to remediate yet — paste in an LLM first.",
                self,
                QtCore.QRect(),
                3500,
            )
            return
        if action == "rephrase":
            self._show_rephrase_card(text)
        elif action == "encrypt":
            self._show_encrypt_card(text)
        elif action == "redact":
            self._show_redact_all_card(text)

    def _show_rephrase_card(self, text: str) -> None:
        self._dismiss_action_card()
        rephrased = mask_pii(text)
        frame = self._ensure_action_card_frame()
        lay = frame.layout()
        if lay is None:
            lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)
        lay.addWidget(QtWidgets.QLabel("Original"))
        orig = QtWidgets.QPlainTextEdit()
        orig.setReadOnly(True)
        orig.setPlainText(text)
        orig.setFixedHeight(72)
        lay.addWidget(orig)
        lay.addWidget(QtWidgets.QLabel("Rephrased (PII removed / replaced)"))
        rep = QtWidgets.QPlainTextEdit()
        rep.setReadOnly(True)
        rep.setPlainText(rephrased)
        rep.setFixedHeight(72)
        lay.addWidget(rep)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self._dismiss_action_card)
        use_btn = QtWidgets.QPushButton("Use this")
        use_btn.setObjectName("primary")

        def _use() -> None:
            QtGui.QGuiApplication.clipboard().setText(rephrased)
            self._dismiss_action_card()

        use_btn.clicked.connect(_use)
        row.addWidget(cancel)
        row.addWidget(use_btn)
        lay.addLayout(row)
        self._mount_action_card(frame)

    def _show_encrypt_card(self, text: str) -> None:
        self._dismiss_action_card()
        hashed = hash_pii(text)
        enc = encrypt_pii(text)
        frame = self._ensure_action_card_frame()
        lay = frame.layout()
        if lay is None:
            lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        lay.addWidget(
            QtWidgets.QLabel("Hashed (PII → HASH::…) and encrypted (PII → ENC::…), demo only")
        )
        body = QtWidgets.QPlainTextEdit()
        body.setReadOnly(True)
        body.setPlainText(f"--- Hashed ---\n{hashed}\n\n--- Encrypted ---\n{enc}")
        body.setFixedHeight(140)
        lay.addWidget(body)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        copy_btn = QtWidgets.QPushButton("Copy encrypted")
        copy_btn.setObjectName("primary")

        def _copy() -> None:
            QtGui.QGuiApplication.clipboard().setText(enc)
            self._dismiss_action_card()

        copy_btn.clicked.connect(_copy)
        row.addWidget(copy_btn)
        lay.addLayout(row)
        self._mount_action_card(frame)

    def _show_redact_all_card(self, text: str) -> None:
        self._dismiss_action_card()
        masked = redact_all_literal(text)
        QtGui.QGuiApplication.clipboard().setText(masked)
        frame = self._ensure_action_card_frame()
        lay = frame.layout()
        if lay is None:
            lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(14, 14, 14, 14)
        msg = QtWidgets.QLabel("Copied to clipboard")
        msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("font-size: 12px; font-weight: bold; color: #5cb85c;")
        lay.addWidget(msg)
        self._mount_action_card(frame)
        self._redact_toast_timer.start(2000)
