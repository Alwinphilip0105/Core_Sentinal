"""
Grammarly-style floating pill for Core Sentinel guardrail.
Visual-only rebuild: pill shape, hover action stack, Win32 input anchoring, state painting.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import List, Optional

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6 import QtCore, QtGui, QtWidgets

from active_window_llm import get_active_llm_name, is_active_window_llm
from pii_remediation import encrypt_pii, hash_pii, mask_pii, redact_all_literal

PILL_W = 72
CARD_W = 280
CARD_SLIDE_PX = 36
PILL_H = 32
PILL_RX = 16
MENU_BTN = 32
MENU_GAP = 8
MENU_SLOT_H = MENU_BTN * 3 + MENU_GAP * 2
PILL_MENU_GAP = 8
# Expanded slide stack: three action buttons + gap + pill (all inside _menu_host)
EXPANDED_MENU_HOST_H = MENU_SLOT_H + PILL_MENU_GAP + PILL_H
STRIP_H = 20
# Hover strip when showing multi-chunk scan subtitle (main + small line)
STRIP_H_MULTI = 34

_MENU_ACTION_TOOLTIPS = {
    "rephrase": "Rephrase — open a card with original vs PII-masked text; “Use this” copies the safe version.",
    "encrypt": "Encrypt — open a card with hashed + ENC:: demo text; “Copy encrypted” copies to the clipboard.",
    "redact": "Redact all — replace detected PII with [REDACTED] and copy to the clipboard.",
}
HOVER_NEAR_PX = 80

# #1a1a1a @ 85% alpha (drawn in local coordinates; multiplied by windowOpacity)
COLOR_PILL_BG = QtGui.QColor(26, 26, 26, int(255 * 0.85))
COLOR_MENU_CIRCLE = QtGui.QColor(26, 26, 26, int(255 * 0.85))
COLOR_SHIELD_ON = QtGui.QColor(21, 195, 154)
COLOR_SHIELD_OFF = QtGui.QColor(120, 122, 128)
COLOR_BADGE = QtGui.QColor(220, 53, 69)
COLOR_CHECK = QtGui.QColor(40, 167, 69)
COLOR_ICON = QtGui.QColor(200, 202, 208)
COLOR_SPINNER_TRACK = QtGui.QColor(60, 62, 66)

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


class _MenuCircleButton(QtWidgets.QWidget):
    """32×32 dark circle with a simple QPainter icon (pencil / lock / eraser)."""

    clicked = QtCore.pyqtSignal()

    def __init__(self, icon: str, parent=None):
        super().__init__(parent)
        self._icon = icon
        self.setFixedSize(MENU_BTN, MENU_BTN)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(COLOR_MENU_CIRCLE)
        p.drawEllipse(0, 0, MENU_BTN, MENU_BTN)
        p.setPen(QtGui.QPen(COLOR_ICON, 1.6))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        c = MENU_BTN // 2
        if self._icon == "pencil":
            p.drawLine(c - 6, c + 6, c + 5, c - 5)
            p.drawLine(c + 2, c - 8, c + 8, c - 2)
            p.drawLine(c - 7, c + 5, c - 5, c + 7)
        elif self._icon == "lock":
            p.drawArc(QtCore.QRectF(c - 5, c - 2, 10, 8), 30 * 16, 120 * 16)
            p.drawRoundedRect(QtCore.QRectF(c - 6, c + 2, 12, 9), 1, 1)
        else:  # eraser
            p.drawRoundedRect(QtCore.QRectF(c - 7, c - 3, 14, 8), 2, 2)
            p.drawLine(c - 5, c + 5, c + 6, c - 4)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class _PillCore(QtWidgets.QWidget):
    """72×32 pill: shield, spinner / check / badge (sits in slide stack with menu)."""

    def __init__(self, controller: "RiskBubble"):
        super().__init__(controller)
        self._c = controller
        self.setFixedSize(PILL_W, PILL_H)
        self.setMouseTracking(True)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._c.mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self._c.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._c.mouseReleaseEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        c = self._c
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        path = QtGui.QPainterPath()
        path.addRoundedRect(0, 0, PILL_W, PILL_H, PILL_RX, PILL_RX)
        p.fillPath(path, COLOR_PILL_BG)

        cx, cy = PILL_W // 2, PILL_H // 2
        shield_fill = COLOR_SHIELD_ON if c._in_llm else COLOR_SHIELD_OFF
        sp = QtGui.QPainterPath()
        sp.moveTo(cx, cy - 9)
        sp.lineTo(cx + 8, cy - 4)
        sp.lineTo(cx + 8, cy + 8)
        sp.lineTo(cx - 8, cy + 8)
        sp.lineTo(cx - 8, cy - 4)
        sp.closeSubpath()
        p.fillPath(sp, shield_fill)
        p.setPen(QtGui.QPen(shield_fill.darker(130), 1))
        p.drawPath(sp)

        scx, scy = PILL_W - 15, PILL_H // 2
        if c._analysing and c._pulse_radius > 0 and c._pulse_ring_alpha > 0.005:
            pr = QtGui.QPen(
                QtGui.QColor(21, 195, 154, int(255 * c._pulse_ring_alpha)), 1.5
            )
            pr.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pr)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawEllipse(
                QtCore.QRectF(scx - c._pulse_radius, scy - c._pulse_radius, 2 * c._pulse_radius, 2 * c._pulse_radius)
            )

        if c._spinner_opacity > 0.01:
            p.save()
            p.setOpacity(c._spinner_opacity)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(COLOR_SPINNER_TRACK)
            p.drawEllipse(QtCore.QRectF(scx - 12, scy - 12, 24, 24))
            pen = QtGui.QPen(COLOR_SHIELD_ON, 2.5)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawArc(
                QtCore.QRectF(scx - 12, scy - 12, 24, 24),
                int(-c._spin_angle * 16),
                int(-270 * 16),
            )
            p.restore()

        if c._show_all_clear and c._all_clear_opacity > 0.01 and not c._issue_count:
            p.save()
            p.setOpacity(c._all_clear_opacity)
            p.setPen(QtGui.QPen(COLOR_CHECK, 2.2))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            rcx, rcy = PILL_W - 14, cy
            chk = QtGui.QPainterPath()
            chk.moveTo(rcx - 5, rcy)
            chk.lineTo(rcx - 1, rcy + 4)
            chk.lineTo(rcx + 6, rcy - 5)
            p.drawPath(chk)
            p.restore()

        if c._issue_count > 0 and not c._analysing:
            badge = str(min(99, c._issue_count))
            bx, by = PILL_W - 17, 1
            p.save()
            p.translate(bx + 7.5, by + 7.5)
            p.scale(c._badge_scale, c._badge_scale)
            p.translate(-7.5, -7.5)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(COLOR_BADGE)
            p.drawEllipse(QtCore.QRectF(0, 0, 15, 15))
            p.setPen(QtGui.QColor(255, 255, 255))
            bf = p.font()
            bf.setBold(True)
            bf.setPointSize(7 if len(badge) > 1 else 8)
            p.setFont(bf)
            p.drawText(QtCore.QRectF(0, 0, 15, 15), QtCore.Qt.AlignmentFlag.AlignCenter, badge)
            p.restore()


class RiskBubble(QtWidgets.QWidget):
    """
    Grammarly-like pill: states, hover menu above, status strip below, Win32 anchor to focused edit.
    """

    menu_action_clicked = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: dict = {}
        self._llm_target = ""
        self._original_text = ""
        self._url = ""
        self._cleaned_title = ""
        self._drag_pos: Optional[QtCore.QPoint] = None
        self._press_pos: Optional[QtCore.QPoint] = None

        self._in_llm = False
        self._analysing = False
        self._hover_ui = False
        self._issue_count = 0
        self._show_all_clear = False
        self._all_clear_opacity = 1.0
        self._menu_expanded = False
        self._menu_anims: List[QtCore.QPropertyAnimation] = []
        self._card_anim: Optional[QtCore.QPropertyAnimation] = None
        self._action_card_inner: Optional[QtWidgets.QWidget] = None
        self._redact_toast_timer = QtCore.QTimer(self)
        self._redact_toast_timer.setSingleShot(True)
        self._redact_toast_timer.timeout.connect(self._dismiss_action_card)

        self._spin_angle = 0
        self._spinner_opacity = 0.0
        self._badge_scale = 1.0
        self._pulse_radius = 0.0
        self._pulse_ring_alpha = 0.0
        self._finish_anim_group: Optional[QtCore.QParallelAnimationGroup] = None
        self._pulse_anim: Optional[QtCore.QVariantAnimation] = None

        self._spinner_timer = QtCore.QTimer(self)
        self._spinner_timer.setInterval(50)
        self._spinner_timer.timeout.connect(self._tick_spinner)

        self._all_clear_timer = QtCore.QTimer(self)
        self._all_clear_timer.setSingleShot(True)
        self._all_clear_timer.timeout.connect(self._begin_all_clear_fade)

        self._fade_timer = QtCore.QTimer(self)
        self._fade_timer.timeout.connect(self._tick_all_clear_fade)

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)

        self._opacity_anim = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        self._opacity_anim.setDuration(180)
        self._opacity_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._action_card_wrap = QtWidgets.QWidget(self)
        self._action_card_wrap.setFixedHeight(0)
        self._action_card_wrap.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        root.addWidget(self._action_card_wrap)

        self._menu_host = QtWidgets.QWidget(self)
        self._menu_host.setFixedHeight(PILL_H)
        self._menu_host.setFixedWidth(PILL_W)
        self._menu_host.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self._menu_hidden_y = EXPANDED_MENU_HOST_H + 4
        self._menu_shown_ys = [8 + i * (MENU_BTN + MENU_GAP) for i in range(3)]
        self._menu_buttons: List[_MenuCircleButton] = []
        _bx0 = (PILL_W - MENU_BTN) // 2
        for key, icon in (("rephrase", "pencil"), ("encrypt", "lock"), ("redact", "eraser")):
            b = _MenuCircleButton(icon, self._menu_host)
            b.setToolTip(_MENU_ACTION_TOOLTIPS.get(key, ""))
            b.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
            b.move(_bx0, -MENU_SLOT_H)
            b.clicked.connect(lambda _=False, k=key: self.menu_action_clicked.emit(k))
            self._menu_buttons.append(b)
        self._pill = _PillCore(self)
        self._pill.setParent(self._menu_host)
        self._pill.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self._pill.raise_()
        root.addWidget(self._menu_host)

        self._hover_label = QtWidgets.QLabel("")
        self._hover_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self._hover_label.setFixedHeight(STRIP_H)
        self._hover_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._hover_label.setStyleSheet(
            "font-family: 'Segoe UI'; "
            "background: rgba(26,26,26,200); border-radius: 6px; padding: 2px 6px;"
        )
        self._hover_label.hide()
        root.addWidget(self._hover_label)

        self._monitor_timer = QtCore.QTimer(self)
        self._monitor_timer.timeout.connect(self._refresh_monitoring)
        self._monitor_timer.start(1000)

        self._anchor_timer = QtCore.QTimer(self)
        self._anchor_timer.timeout.connect(self._refresh_anchor_position)
        self._anchor_timer.start(1000)

        self._refresh_monitoring()
        self._resize_to_state()
        self._apply_opacity_target(animate=False)
        self.setToolTip("Core Sentinel — drag to move; click for remediation when issues exist")

    def _spinnerOpacity(self) -> float:
        return self._spinner_opacity

    def _set_spinnerOpacity(self, v: float) -> None:
        self._spinner_opacity = float(v)
        self._pill.update()

    spinnerOpacity = QtCore.pyqtProperty(float, _spinnerOpacity, _set_spinnerOpacity)

    def _allClearOpacity(self) -> float:
        return self._all_clear_opacity

    def _set_allClearOpacity(self, v: float) -> None:
        self._all_clear_opacity = float(v)
        self._pill.update()

    allClearOpacity = QtCore.pyqtProperty(float, _allClearOpacity, _set_allClearOpacity)

    def _badgeScale(self) -> float:
        return self._badge_scale

    def _set_badgeScale(self, v: float) -> None:
        self._badge_scale = float(v)
        self._pill.update()

    badgeScale = QtCore.pyqtProperty(float, _badgeScale, _set_badgeScale)

    def _menu_btn_x(self) -> int:
        return max(0, (self._menu_host.width() - MENU_BTN) // 2)

    def _pill_center_offset_from_window_top(self) -> int:
        # Pill is anchored at the bottom of _menu_host
        return self._action_card_wrap.height() + self._menu_host.height() - PILL_H // 2

    def _hover_strip_height(self) -> int:
        if not self._hover_ui:
            return 0
        r = self._result if isinstance(self._result, dict) else {}
        if (
            not self._analysing
            and bool(r.get("text_truncated"))
            and int(r.get("chunks_scored") or 0) > 1
        ):
            return STRIP_H_MULTI
        return STRIP_H

    def _resize_to_state(self) -> None:
        w = CARD_W if self._action_card_wrap.height() > 0 else PILL_W
        self._menu_host.setFixedWidth(w)
        h = self._action_card_wrap.height() + self._menu_host.height() + self._hover_strip_height()
        self.setFixedSize(w, max(h, PILL_H))
        if not self._menu_anims:
            self._sync_menu_button_positions()

    def _layout_pill_in_menu_host(self) -> None:
        w = self._menu_host.width()
        h = self._menu_host.height()
        x = max(0, (w - PILL_W) // 2)
        y = max(0, h - PILL_H)
        self._pill.move(x, y)

    def _sync_menu_button_positions(self) -> None:
        bx = self._menu_btn_x()
        if self._menu_host.height() <= PILL_H:
            for b in self._menu_buttons:
                b.move(bx, -MENU_SLOT_H + 8)
        else:
            for i, b in enumerate(self._menu_buttons):
                b.move(bx, self._menu_shown_ys[i])
        self._layout_pill_in_menu_host()

    def _base_window_opacity(self) -> float:
        if not self._in_llm:
            return 0.5
        return 0.7

    def _apply_opacity_target(self, *, animate: bool = True) -> None:
        target = 1.0 if self._hover_ui else self._base_window_opacity()
        if not animate:
            self._opacity_anim.stop()
            self.setWindowOpacity(target)
            return
        if self._opacity_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self._opacity_anim.stop()
        self._opacity_anim.setStartValue(self.windowOpacity())
        self._opacity_anim.setEndValue(target)
        self._opacity_anim.start()

    def _hover_status_main_line(self) -> str:
        if not self._in_llm:
            return "Not monitoring"
        if self._analysing:
            return get_active_llm_name() or self._llm_target or "LLM"
        if self._issue_count > 0:
            n = self._issue_count
            return f"{n} issue{'s' if n != 1 else ''} found"
        return get_active_llm_name() or self._llm_target or "LLM"

    def _hover_status_text(self) -> str:
        """Plain single-line status (main line only)."""
        return self._hover_status_main_line()

    def _hover_label_html(self) -> str:
        """Rich text for hover strip: optional second line when long text was multi-chunk scanned."""
        main = html.escape(self._hover_status_main_line())
        body = (
            f'<div align="center" style="line-height:1.2;">'
            f'<span style="font-size:10px;color:#c8c8c8;">{main}</span>'
        )
        r = self._result if isinstance(self._result, dict) else {}
        if (
            not self._analysing
            and bool(r.get("text_truncated"))
            and int(r.get("chunks_scored") or 0) > 1
        ):
            llm = html.escape(get_active_llm_name() or self._llm_target or "LLM")
            n = int(r.get("chunks_scored") or 0)
            sub = html.escape(f"{llm} · {n} chunks scanned")
            body += f'<br/><span style="font-size:8px;color:#909090;">{sub}</span>'
        body += "</div>"
        return body

    def _update_hover_label(self) -> None:
        if self._hover_ui:
            self._hover_label.setText(self._hover_label_html())
            self._hover_label.setFixedHeight(self._hover_strip_height())
            self._hover_label.show()
        else:
            self._hover_label.setText("")
            self._hover_label.setFixedHeight(STRIP_H)
            self._hover_label.hide()
        self._resize_to_state()

    def _stop_menu_anims(self) -> None:
        for a in self._menu_anims:
            a.stop()
        self._menu_anims.clear()

    def _pill_center_global(self) -> QtCore.QPoint:
        return self._pill.mapToGlobal(QtCore.QPoint(PILL_W // 2, PILL_H // 2))

    def _set_menu_geometry_expanded(self, expand: bool) -> None:
        pill_c_desired = self._pill_center_global().y()
        if expand:
            self._menu_host.setFixedHeight(EXPANDED_MENU_HOST_H)
        else:
            self._menu_host.setFixedHeight(PILL_H)
        new_off = self._pill_center_offset_from_window_top()
        new_y = int(pill_c_desired - new_off)
        self.move(self.x(), new_y)
        self._resize_to_state()
        self._sync_menu_button_positions()

    def _run_menu_show_anims(self) -> None:
        self._stop_menu_anims()
        btn_x = self._menu_btn_x()
        for i, b in enumerate(self._menu_buttons):
            end = QtCore.QPoint(btn_x, self._menu_shown_ys[i])
            start = QtCore.QPoint(btn_x, self._menu_hidden_y)
            b.move(start)
            anim = QtCore.QPropertyAnimation(b, b"pos", self)
            anim.setDuration(220)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
            anim.setStartValue(start)
            anim.setEndValue(end)
            self._menu_anims.append(anim)
            QtCore.QTimer.singleShot(i * 50, lambda a=anim: a.start())

    def _run_menu_hide_anims(self) -> None:
        self._stop_menu_anims()
        btn_x = self._menu_btn_x()
        end = QtCore.QPoint(btn_x, self._menu_hidden_y)

        def collapse_after():
            if self._hover_ui:
                return
            self._menu_expanded = False
            self._set_menu_geometry_expanded(False)

        n = len(self._menu_buttons)
        for i, b in enumerate(self._menu_buttons):
            start = b.pos()
            anim = QtCore.QPropertyAnimation(b, b"pos", self)
            anim.setDuration(200)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
            anim.setStartValue(start)
            anim.setEndValue(end)
            self._menu_anims.append(anim)
            delay_ms = (n - 1 - i) * 50
            if i == 0:
                anim.finished.connect(collapse_after)
            QtCore.QTimer.singleShot(delay_ms, lambda a=anim: a.start())

    def _set_hover_true(self) -> None:
        if self._hover_ui:
            return
        self._hover_ui = True
        self._stop_menu_anims()
        if self._menu_host.height() <= PILL_H:
            self._menu_expanded = True
            self._set_menu_geometry_expanded(True)
        self._run_menu_show_anims()
        self._update_hover_label()
        self._apply_opacity_target()
        self._pill.update()

    def _set_hover_false(self) -> None:
        if not self._hover_ui:
            return
        self._hover_ui = False
        self._update_hover_label()
        self._apply_opacity_target()
        if self._menu_expanded:
            self._run_menu_hide_anims()
        else:
            self._pill.update()

    def _defer_hover_leave_check(self) -> None:
        w = QtWidgets.QApplication.widgetAt(QtGui.QCursor.pos())
        if w is None:
            self._set_hover_false()
            return
        if w is self or self.isAncestorOf(w):
            return
        self._set_hover_false()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() in (
            QtCore.QEvent.Type.Enter,
            QtCore.QEvent.Type.HoverEnter,
        ):
            if obj is self or self.isAncestorOf(obj) or obj in self._menu_buttons:
                self._set_hover_true()
        elif event.type() in (
            QtCore.QEvent.Type.Leave,
            QtCore.QEvent.Type.HoverLeave,
        ):
            QtCore.QTimer.singleShot(0, self._defer_hover_leave_check)
        return super().eventFilter(obj, event)

    def enterEvent(self, event: QtCore.QEvent) -> None:
        self._set_hover_true()
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        QtCore.QTimer.singleShot(0, self._defer_hover_leave_check)
        super().leaveEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.installEventFilter(self)
        self._menu_host.installEventFilter(self)
        self._pill.installEventFilter(self)
        self._hover_label.installEventFilter(self)
        self._action_card_wrap.installEventFilter(self)
        for b in self._menu_buttons:
            b.installEventFilter(self)

    def _refresh_monitoring(self) -> None:
        was = self._in_llm
        self._in_llm = bool(get_active_llm_name())
        if was != self._in_llm:
            if not self._in_llm:
                self.set_idle()
        self._pill.update()
        if self._hover_ui:
            self._update_hover_label()
        elif was != self._in_llm:
            self._apply_opacity_target()

    def _refresh_anchor_position(self) -> None:
        if self._drag_pos is not None:
            return
        try:
            self._refresh_anchor_position_impl()
        except Exception:
            pass

    def _refresh_anchor_position_impl(self) -> None:
        is_llm, _, _, _ = is_active_window_llm()
        if not is_llm:
            self.show_near_bottom_right()
            return

        left = top = right = bottom = None
        try:
            from input_field_rect import get_input_field_rect

            ir = get_input_field_rect()
            if ir is not None:
                ix, iy, iw, ih = ir
                left, top, right, bottom = ix, iy, ix + iw, iy + ih
        except ImportError:
            pass
        except Exception:
            pass

        if left is None:
            try:
                from win_input_anchor import get_focused_text_surface_rect_screen

                r = get_focused_text_surface_rect_screen()
                if r is None:
                    return
                left, top, right, bottom = r
            except ImportError:
                return
            except Exception:
                return

        cur = QtGui.QCursor.pos()
        exp = QtCore.QRect(
            left - HOVER_NEAR_PX,
            top - HOVER_NEAR_PX,
            (right - left) + 2 * HOVER_NEAR_PX,
            (bottom - top) + 2 * HOVER_NEAR_PX,
        )
        if not exp.contains(cur):
            return

        pill_c_desired_y = (top + bottom) // 2
        pill_c_desired_x = right + 6 + PILL_W // 2
        off = self._pill_center_offset_from_window_top()
        x = pill_c_desired_x - PILL_W // 2
        y = pill_c_desired_y - off
        self.move(int(x), int(y))

    def _stop_finish_anims(self) -> None:
        if self._finish_anim_group is not None:
            self._finish_anim_group.stop()
            self._finish_anim_group.deleteLater()
            self._finish_anim_group = None

    def _stop_pulse_anim(self) -> None:
        if self._pulse_anim is not None:
            self._pulse_anim.stop()
            self._pulse_anim.deleteLater()
            self._pulse_anim = None
        self._pulse_radius = 0.0
        self._pulse_ring_alpha = 0.0

    def _on_pulse_frame(self, value: object) -> None:
        t = float(value)
        self._pulse_radius = 18.0 + 10.0 * t
        self._pulse_ring_alpha = 0.4 * (1.0 - t)
        self._pill.update()

    def _start_pulse_anim(self) -> None:
        self._stop_pulse_anim()
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(1200)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setLoopCount(-1)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.Linear)
        anim.valueChanged.connect(self._on_pulse_frame)
        self._pulse_anim = anim
        anim.start()

    def set_analysing(self) -> None:
        self._stop_finish_anims()
        self._all_clear_timer.stop()
        self._fade_timer.stop()
        self._show_all_clear = False
        self.setAllClearOpacity(0.0)
        self._issue_count = 0
        self.setBadgeScale(1.0)
        self._analysing = True
        self._spin_angle = 0
        self.setSpinnerOpacity(1.0)
        self._spinner_timer.start()
        self._start_pulse_anim()
        self._pill.update()
        if self._hover_ui:
            self._update_hover_label()

    def set_idle(self) -> None:
        self._stop_finish_anims()
        self._stop_pulse_anim()
        self._spinner_timer.stop()
        self._analysing = False
        self.setSpinnerOpacity(0.0)
        self._show_all_clear = False
        self.setAllClearOpacity(0.0)
        self._issue_count = 0
        self.setBadgeScale(1.0)
        self._all_clear_timer.stop()
        self._fade_timer.stop()
        self._pill.update()
        if self._hover_ui:
            self._update_hover_label()

    def set_clear(self) -> None:
        self._stop_pulse_anim()
        self._spinner_timer.stop()
        self._analysing = False
        self._issue_count = 0
        self._show_all_clear = True
        self._all_clear_timer.stop()
        self._fade_timer.stop()
        self._stop_finish_anims()

        self.setAllClearOpacity(0.0)
        self.setSpinnerOpacity(1.0)

        group = QtCore.QParallelAnimationGroup(self)
        a_spin = QtCore.QPropertyAnimation(self, b"spinnerOpacity", self)
        a_spin.setDuration(200)
        a_spin.setStartValue(1.0)
        a_spin.setEndValue(0.0)
        a_spin.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        a_chk = QtCore.QPropertyAnimation(self, b"allClearOpacity", self)
        a_chk.setDuration(200)
        a_chk.setStartValue(0.0)
        a_chk.setEndValue(1.0)
        a_chk.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        group.addAnimation(a_spin)
        group.addAnimation(a_chk)

        def _on_clear_finish() -> None:
            self._finish_anim_group = None
            if self._all_clear_timer.isActive():
                self._all_clear_timer.stop()
            self._all_clear_timer.start(3000)

        group.finished.connect(_on_clear_finish)
        self._finish_anim_group = group
        group.start()
        self._pill.update()
        if self._hover_ui:
            self._update_hover_label()

    def set_issues(self, count: int) -> None:
        n = max(0, int(count))
        if n <= 0:
            self.set_clear()
            return

        self._stop_pulse_anim()
        self._spinner_timer.stop()
        self._analysing = False
        self._show_all_clear = False
        self.setAllClearOpacity(0.0)
        self._all_clear_timer.stop()
        self._fade_timer.stop()
        self._stop_finish_anims()

        self._issue_count = n
        self.setSpinnerOpacity(1.0)
        self.setBadgeScale(0.5)

        group = QtCore.QParallelAnimationGroup(self)
        a_spin = QtCore.QPropertyAnimation(self, b"spinnerOpacity", self)
        a_spin.setDuration(200)
        a_spin.setStartValue(1.0)
        a_spin.setEndValue(0.0)
        a_spin.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        a_badge = QtCore.QPropertyAnimation(self, b"badgeScale", self)
        a_badge.setDuration(200)
        a_badge.setStartValue(0.5)
        a_badge.setEndValue(1.0)
        a_badge.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        group.addAnimation(a_spin)
        group.addAnimation(a_badge)

        def _on_issues_finish() -> None:
            self._finish_anim_group = None

        group.finished.connect(_on_issues_finish)
        self._finish_anim_group = group
        group.start()
        self._pill.update()
        if self._hover_ui:
            self._update_hover_label()

    def setSpinnerOpacity(self, v: float) -> None:
        self._set_spinnerOpacity(v)

    def setAllClearOpacity(self, v: float) -> None:
        self._set_allClearOpacity(v)

    def setBadgeScale(self, v: float) -> None:
        self._set_badgeScale(v)

    def _tick_spinner(self) -> None:
        if not self._analysing:
            return
        self._spin_angle = (self._spin_angle + 18) % 360
        self._pill.update()

    def _begin_all_clear_fade(self) -> None:
        self._fade_timer.start(40)

    def _tick_all_clear_fade(self) -> None:
        self._all_clear_opacity -= 0.08
        if self._all_clear_opacity <= 0:
            self._all_clear_opacity = 0.0
            self._show_all_clear = False
            self._fade_timer.stop()
        self._pill.update()
        if self._hover_ui:
            self._update_hover_label()

    def clear_issue_badge(self) -> None:
        self._issue_count = 0
        self._pill.update()
        if self._hover_ui:
            self._update_hover_label()

    def update_from_result(
        self,
        result: dict,
        llm_target: str,
        original_text: str = "",
        url: Optional[str] = None,
        cleaned_title: Optional[str] = None,
    ) -> None:
        r: dict = result if isinstance(result, dict) else {}
        self._result = r
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
        try:
            score = int(r.get("risk_score", 0))
        except (TypeError, ValueError):
            score = 0
        critical = bool(r.get("critical_secret_detected", False))

        if action == "silent" and not critical and score < 30 and not triggers:
            self.set_clear()
        else:
            n = len(triggers)
            if n <= 0 and (action in ("warn", "block") or critical or score >= 30):
                n = 1
            self.set_issues(n)

        risk = r.get("risk", "low")
        tip = f"PII risk {score}/100 ({risk}) — {llm_target}"
        if self._cleaned_title:
            tip += f" — {self._cleaned_title}"
        self.setToolTip(tip)
        self._pill.update()
        if self._hover_ui:
            self._update_hover_label()

    def show_near_bottom_right(self) -> None:
        screen = QtGui.QGuiApplication.primaryScreen()
        if not screen:
            return
        geom = screen.availableGeometry()
        m = 24
        x = geom.right() - self.width() - m
        y = geom.bottom() - self.height() - m
        self.move(x, y)

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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._press_pos = event.globalPosition().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_pos is not None and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._drag_pos is not None:
            moved = (event.globalPosition().toPoint() - self._press_pos).manhattanLength()
            self._drag_pos = None
            self._press_pos = None
            if moved < 5:
                if self._issue_count > 0 or int(self._result.get("risk_score", 0)) > 0:
                    from ui_remediation_dialog import RemediationDialog

                    self.hide()
                    dialog = RemediationDialog(
                        self._original_text,
                        self._result,
                        self._llm_target,
                        self,
                    )
                    dialog.exec()
                    self.clear_issue_badge()
                    self.show()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _clear_action_card_children(self) -> None:
        if self._card_anim:
            self._card_anim.stop()
            self._card_anim = None
        for ch in list(self._action_card_wrap.children()):
            if isinstance(ch, QtWidgets.QWidget):
                ch.setParent(None)
                ch.deleteLater()
        self._action_card_inner = None

    def _dismiss_action_card(self) -> None:
        self._redact_toast_timer.stop()
        if self._action_card_wrap.height() <= 0:
            self._clear_action_card_children()
            return
        pill_cy = self._pill_center_global().y()
        self._clear_action_card_children()
        self._action_card_wrap.setFixedHeight(0)
        self._resize_to_state()
        self.move(self.x(), int(pill_cy - self._pill_center_offset_from_window_top()))

    def _slide_in_card(self, frame: QtWidgets.QWidget) -> None:
        if self._card_anim:
            self._card_anim.stop()
        self._card_anim = QtCore.QPropertyAnimation(frame, b"pos", self)
        self._card_anim.setDuration(220)
        self._card_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._card_anim.setStartValue(frame.pos())
        self._card_anim.setEndValue(QtCore.QPoint(frame.x(), 4))
        self._card_anim.start()

    def _mount_action_card(self, frame: QtWidgets.QWidget) -> None:
        self._clear_action_card_children()
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
        self._action_card_wrap.setFixedHeight(wrap_h)
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
        frame = QtWidgets.QFrame()
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
        frame = QtWidgets.QFrame()
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
        redacted = redact_all_literal(text)
        QtGui.QGuiApplication.clipboard().setText(redacted)
        frame = QtWidgets.QFrame()
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(14, 14, 14, 14)
        msg = QtWidgets.QLabel("Copied to clipboard")
        msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("font-size: 12px; font-weight: bold; color: #5cb85c;")
        lay.addWidget(msg)
        self._mount_action_card(frame)
        self._redact_toast_timer.start(2000)
