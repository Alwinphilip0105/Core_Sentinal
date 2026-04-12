"""
Behavioral character + speech bubble near the traffic light after scored pastes.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets


class _GlobalClickFilter(QtCore.QObject):
    """Hide the character when the user clicks anywhere (including other windows)."""

    def __init__(self, target: "CharacterWidget") -> None:
        super().__init__(target)
        self._target = target

    def eventFilter(self, obj: Optional[QtCore.QObject], event: Optional[QtCore.QEvent]) -> bool:
        if event is None:
            return False
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            self._target.hide()
        return False


class _TriangleWidget(QtWidgets.QWidget):
    """Small upward-pointing triangle (speech bubble tail toward emoji)."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtGui.QPen(QtGui.QColor(0xAA, 0xAA, 0xAA), 1.0))
        p.setBrush(QtGui.QColor(255, 255, 255))
        w = self.width()
        h = self.height()
        cx = w // 2
        tri_w = 14
        path = QtGui.QPainterPath()
        path.moveTo(float(cx - tri_w // 2), float(h))
        path.lineTo(float(cx), 0.0)
        path.lineTo(float(cx + tri_w // 2), float(h))
        path.closeSubpath()
        p.drawPath(path)


class _BubbleLabel(QtWidgets.QLabel):
    """Rounded white bubble with 1px grey border; text color from palette."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter
        )
        self.setMinimumWidth(104)
        self.setMaximumWidth(112)
        self.setMargin(8)


class CharacterWidget(QtWidgets.QWidget):
    """
    120px wide, auto height: emoji row + triangle + speech bubble.
    Fade-in 300ms; auto-hide 6s; global click hides.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(
            parent,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedWidth(120)

        self._emoji = QtWidgets.QLabel()
        self._emoji.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._emoji.setFixedSize(120, 44)
        f = QtGui.QFont(self._emoji.font())
        f.setPixelSize(40)
        self._emoji.setFont(f)

        self._badge = QtWidgets.QLabel()
        self._badge.setStyleSheet(
            "background:#E53935;color:white;border-radius:9px;padding:2px 6px;"
            "font-size:10px;font-weight:600;font-family:'Segoe UI';"
        )
        self._badge.hide()

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        top.addStretch(1)
        top.addWidget(self._emoji, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self._badge, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        top.addStretch(1)

        self._triangle = _TriangleWidget()
        self._triangle.setFixedSize(120, 10)

        self._bubble = _BubbleLabel()
        _bf = QtGui.QFont()
        _bf.setFamily("Segoe UI")
        _bf.setPixelSize(12)
        self._bubble.setFont(_bf)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addLayout(top)
        root.addWidget(self._triangle)
        root.addWidget(self._bubble)

        self._opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(0.0)

        self._fade_anim: Optional[QtCore.QPropertyAnimation] = None
        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self._click_filter: Optional[_GlobalClickFilter] = None
        self._filtered_widgets: list[QtWidgets.QWidget] = []

    def configure(
        self,
        emoji: str,
        bubble_text: str,
        text_color: str,
        *,
        large_text: bool = False,
        streak_badge: Optional[str] = None,
    ) -> None:
        self._emoji.setText(emoji)
        self._bubble.setText(bubble_text)
        self._bubble.setStyleSheet(
            f"color: {text_color}; background: white; border: 1px solid #AAAAAA; "
            f"border-radius: 10px; font-family: 'Segoe UI'; font-size: {14 if large_text else 12}px;"
        )
        if streak_badge:
            self._badge.setText(streak_badge)
            self._badge.show()
        else:
            self._badge.hide()

    def show_near_traffic_light(self, traffic_light: QtWidgets.QWidget) -> None:
        self._hide_timer.stop()
        if self._fade_anim is not None:
            self._fade_anim.stop()
        self._uninstall_global_click()

        self.adjustSize()
        screen = QtWidgets.QApplication.primaryScreen()
        g = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1920, 1080)
        x = traffic_light.x() + traffic_light.width() + 12
        if x + 140 > g.right():
            x = traffic_light.x() - 152
        y = traffic_light.y() + traffic_light.height() // 2 - self.height() // 2
        y = max(g.top(), min(y, g.bottom() - self.height()))
        x = max(g.left(), min(x, g.right() - self.width()))
        self.move(x, y)

        self.raise_()
        self.show()

        self._opacity_effect.setOpacity(0.0)
        self._fade_anim = QtCore.QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setDuration(300)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

        self._hide_timer.start(6000)

        app = QtWidgets.QApplication.instance()
        if app is not None:
            self._click_filter = _GlobalClickFilter(self)
            for w in app.allWidgets():
                w.installEventFilter(self._click_filter)
                self._filtered_widgets.append(w)

    def _uninstall_global_click(self) -> None:
        if self._click_filter is not None:
            for w in self._filtered_widgets:
                try:
                    w.removeEventFilter(self._click_filter)
                except RuntimeError:
                    pass
            self._filtered_widgets.clear()
        self._click_filter = None

    def hideEvent(self, event: QtGui.QEvent) -> None:
        self._hide_timer.stop()
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim = None
        self._uninstall_global_click()
        super().hideEvent(event)
