"""
Separate traffic-light widget: sits flush left of the RiskBubble pill, 28×52.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QToolTip

W_TL = 28
H_TL = 52

COLOR_HOUSING = QtGui.QColor(28, 28, 30, 215)
COLOR_BORDER = QtGui.QColor(200, 200, 200, 80)


class TrafficLightIndicator(QtWidgets.QWidget):
    """28×52; left-rounded housing; flush to pill’s left edge (no horizontal gap)."""

    remediation_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        pill: QtWidgets.QWidget,
        pill_rect: QtCore.QRect | None = None,
    ) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowStaysOnTopHint,
        )
        self._pill = pill
        # Region of `pill` that draws the fused chrome (defaults to full widget).
        self._pill_rect = (
            pill_rect
            if pill_rect is not None
            else QtCore.QRect(0, 0, max(1, pill.width()), max(1, pill.height()))
        )
        self.setFixedSize(W_TL, H_TL)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAutoFillBackground(False)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        self._state = ""
        self._pulse = True
        self._pulse_timer = QtCore.QTimer(self)
        self._pulse_timer.setInterval(600)
        self._pulse_timer.timeout.connect(self._tick_pulse)

        self._pill.installEventFilter(self)
        self.setMouseTracking(True)

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
        path = QtGui.QPainterPath()
        path.moveTo(8.5, 0.5)
        path.lineTo(27.5, 0.5)
        path.lineTo(27.5, 51.5)
        path.lineTo(8.5, 51.5)
        path.arcTo(QtCore.QRectF(0.5, 35.5, 16.0, 16.0), 270, -90)
        path.lineTo(0.5, 8.5)
        path.arcTo(QtCore.QRectF(0.5, 0.5, 16.0, 16.0), 180, -90)
        path.closeSubpath()
        return path

    def _border_path(self) -> QtGui.QPainterPath:
        border = QtGui.QPainterPath()
        border.moveTo(27.5, 0.5)
        border.lineTo(8.5, 0.5)
        border.arcTo(QtCore.QRectF(0.5, 0.5, 16.0, 16.0), 90, 90)
        border.lineTo(0.5, 35.5)
        border.arcTo(QtCore.QRectF(0.5, 35.5, 16.0, 16.0), 180, 90)
        border.lineTo(27.5, 51.5)
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
        self._sync_opacity_from_pill()
        self._reposition()

    def _sync_opacity_from_pill(self) -> None:
        if self._pill is not None:
            self.setWindowOpacity(self._pill.windowOpacity())

    def _reposition(self) -> None:
        if not self._pill:
            return
        top_left = self._pill.mapToGlobal(
            QtCore.QPoint(self._pill_rect.left(), self._pill_rect.top())
        )
        x = top_left.x() - W_TL
        y = top_left.y() + (self._pill_rect.height() - H_TL) // 2
        center = QtCore.QPoint(x + W_TL // 2, y + H_TL // 2)
        app = QtGui.QGuiApplication.instance()
        screen = app.screenAt(center) if app else None
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        g = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1920, 1080)
        x = max(g.left() + 4, min(x, g.right() - W_TL - 3))
        y = max(g.top() + 4, min(y, g.bottom() - H_TL - 4))
        self.move(x, y)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        fill_path = self._housing_fill_path()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(COLOR_HOUSING)
        painter.drawPath(fill_path)

        painter.setPen(QtGui.QPen(COLOR_BORDER, 1.0))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(self._border_path())
        # Vertical seam between traffic light and pill (square right edge of housing)
        painter.drawLine(QtCore.QLineF(27.5, 0.5, 27.5, 51.5))

        dot_r = 5.0
        cx = 14.0
        dot_data = [
            (10.0, QtGui.QColor("#E53935")),
            (26.0, QtGui.QColor("#FFB300")),
            (42.0, QtGui.QColor("#43A047")),
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
                painter.drawEllipse(pt, dot_r + 4.0, dot_r + 4.0)
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
