"""
Traffic-light segment fused with the RiskBubble pill (single component chrome).
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QToolTip

W_TL = 28
# Match the pill body height so both segments share the same top/bottom boundary.
H_TL = 34

COLOR_HOUSING = QtGui.QColor(28, 28, 30, 215)
COLOR_BORDER = QtGui.QColor(200, 200, 200, 80)


class TrafficLightIndicator(QtWidgets.QWidget):
    """Left-rounded traffic segment; flush to pill’s left edge with shared boundaries."""

    remediation_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        pill: QtWidgets.QWidget,
        pill_rect: QtCore.QRect | None = None,
    ) -> None:
        # Child of RiskBubble — one unified window with the pill (no separate Tool window).
        super().__init__(pill)
        self._pill = pill
        # Region of `pill` that draws the fused chrome (defaults to full widget).
        self._pill_rect = (
            pill_rect
            if pill_rect is not None
            else QtCore.QRect(0, 0, max(1, pill.width()), max(1, pill.height()))
        )
        self.setFixedSize(W_TL, H_TL)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        self._state = ""
        self._pulse = True
        self._pulse_timer = QtCore.QTimer(self)
        self._pulse_timer.setInterval(600)
        self._pulse_timer.timeout.connect(self._tick_pulse)

        self._pill.installEventFilter(self)
        self.setMouseTracking(True)
        self.raise_()

    def _get_tooltip_text(self) -> str:
        state = str(getattr(self, "_state", "idle") or "idle").lower()
        pill = self._pill
        score = int(getattr(pill, "_preview_score", 0) or 0) if pill is not None else 0
        prev = bool(getattr(pill, "_clipboard_preview_active", False)) if pill is not None else False
        if prev and state in ("high", "med", "safe"):
            if state == "high":
                return (
                    f"⚠ Clipboard contains HIGH risk content (score {score}) — "
                    f"will be blocked on paste"
                )
            if state == "med":
                return (
                    f"⚠ Clipboard contains medium risk content (score {score}) — "
                    f"will be reviewed on paste"
                )
            return f"✓ Clipboard is safe (score {score}) — will paste normally"
        msgs = {
            "high": f"⚠ High risk (score {score})" if score else "⚠ High risk",
            "med": f"⚠ Medium risk (score {score})" if score else "⚠ Medium risk",
            "safe": f"✓ Safe (score {score})" if score else "✓ Safe",
            "idle": "Sentinel monitoring clipboard…",
            "not_monitoring": "Not monitoring this context — open an LLM tab or resume monitoring",
            "analysing": "Scoring paste…",
        }
        return msgs.get(state, "Sentinel active")

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        QToolTip.showText(
            self.mapToGlobal(event.position().toPoint()),
            self._get_tooltip_text(),
            self,
        )
        super().mouseMoveEvent(event)

    def set_pill_rect(self, rect: QtCore.QRect) -> None:
        self._pill_rect = rect
        self._reposition()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self._pill:
            if event.type() in (
                QtCore.QEvent.Type.Move,
                QtCore.QEvent.Type.Show,
                QtCore.QEvent.Type.Resize,
            ):
                self._reposition()
        return False

    def _housing_fill_path(self) -> QtGui.QPainterPath:
        right = float(W_TL) - 0.5
        bottom = float(H_TL) - 0.5
        arc_h = 16.0
        arc_top_y = 0.5
        arc_bottom_y = bottom - arc_h + 1.0
        path = QtGui.QPainterPath()
        path.moveTo(8.5, 0.5)
        path.lineTo(right, 0.5)
        path.lineTo(right, bottom)
        path.lineTo(8.5, bottom)
        path.arcTo(QtCore.QRectF(0.5, arc_bottom_y, 16.0, arc_h), 270, -90)
        path.lineTo(0.5, 8.5)
        path.arcTo(QtCore.QRectF(0.5, arc_top_y, 16.0, arc_h), 180, -90)
        path.closeSubpath()
        return path

    def _border_path(self) -> QtGui.QPainterPath:
        right = float(W_TL) - 0.5
        bottom = float(H_TL) - 0.5
        arc_h = 16.0
        arc_top_y = 0.5
        arc_bottom_y = bottom - arc_h + 1.0
        border = QtGui.QPainterPath()
        border.moveTo(right, 0.5)
        border.lineTo(8.5, 0.5)
        border.arcTo(QtCore.QRectF(0.5, arc_top_y, 16.0, arc_h), 90, 90)
        border.lineTo(0.5, bottom - 8.0)
        border.arcTo(QtCore.QRectF(0.5, arc_bottom_y, 16.0, arc_h), 180, 90)
        border.lineTo(right, bottom)
        return border

    def _active_index(self) -> int:
        s = str(getattr(self, "_state", "") or "").lower()
        return {
            "high": 0,
            "block": 0,
            "med": 1,
            "warn": 1,
            "analysing": 1,
            "safe": 2,
            "clear": 2,
            # Monitoring LLM, no score yet — same “ready / OK” lane as safe
            "idle": 2,
        }.get(s, -1)

    def set_state(self, state: str) -> None:
        self._state = str(state or "").lower().strip()
        if self._state == "analysing":
            self._pulse = True
            self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
            self._pulse = True
        self.update()
        self._reposition()

    def _tick_pulse(self) -> None:
        if self._state != "analysing":
            self._pulse_timer.stop()
            self._pulse = True
            self.update()
            return
        self._pulse = not self._pulse
        self.update()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._reposition()

    def _reposition(self) -> None:
        if not self._pill:
            return
        # Local coords: flush left, vertically centered on the pill body rect.
        y = self._pill_rect.top() + (self._pill_rect.height() - H_TL) // 2
        self.move(0, int(y))
        self.raise_()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        fill_path = self._housing_fill_path()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(COLOR_HOUSING)
        painter.drawPath(fill_path)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        # Do not draw a separate seam stroke here; the pill border handles the join.
        # This avoids a doubled/thicker border where the two segments meet.

        # Smaller dots to reduce visual clutter in the compact 34px column.
        dot_r = 3.6
        cx = 14.0
        margin_y = 7.0
        spacing = (max(1.0, float(H_TL) - (2.0 * margin_y))) / 2.0
        dot_data = [
            (margin_y, QtGui.QColor("#E53935")),
            (margin_y + spacing, QtGui.QColor("#FFB300")),
            (margin_y + (2.0 * spacing), QtGui.QColor("#43A047")),
        ]
        active = self._active_index()

        for i, (cy, color) in enumerate(dot_data):
            pt = QtCore.QPointF(cx, cy)
            if self._state == "analysing" and i == 1:
                if not self._pulse:
                    dim = QtGui.QColor(color)
                    dim.setAlpha(28)
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.setBrush(dim)
                    painter.drawEllipse(pt, dot_r, dot_r)
                    continue
            if i == active:
                glow = QtGui.QColor(color)
                glow.setAlpha(50)
                painter.setBrush(glow)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawEllipse(pt, dot_r + 2.4, dot_r + 2.4)
                painter.setBrush(color)
                painter.drawEllipse(pt, dot_r, dot_r)
            else:
                dim = QtGui.QColor(color)
                dim.setAlpha(28)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(dim)
                painter.drawEllipse(pt, dot_r, dot_r)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.remediation_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)
