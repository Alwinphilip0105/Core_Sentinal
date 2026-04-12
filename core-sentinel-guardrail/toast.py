"""Non-modal bottom-right toast notifications (shared by main + bubble toolbar)."""

from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRect, QTimer, Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class Toast(QWidget):
    """Non-modal bottom-right toast with fade in/out."""

    def __init__(
        self,
        message: str,
        parent=None,
        color: str = "#02C39A",
        duration: int = 3000,
        title: str | None = None,
    ):
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 10, 16, 10)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {color}; font-size: 10px;")
        dot.setAlignment(Qt.AlignmentFlag.AlignTop)
        outer.addWidget(dot, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)
        if title:
            tl = QLabel(title)
            tl.setWordWrap(True)
            tl.setMaximumWidth(420)
            tl.setStyleSheet(
                "color: rgba(255,255,255,0.95); font-size: 12px; "
                "font-weight: 700; background: transparent;"
            )
            text_col.addWidget(tl)
        lbl = QLabel(message)
        lbl.setWordWrap(True)
        lbl.setMaximumWidth(420)
        lbl.setStyleSheet(
            "color: white; font-size: 12px; "
            "font-weight: 500; background: transparent;"
        )
        text_col.addWidget(lbl)
        outer.addLayout(text_col)

        self.setStyleSheet(
            "background-color: rgba(28,28,30,240);"
            "border-radius: 10px;"
            "border: 1px solid rgba(255,255,255,0.1);"
        )
        self.adjustSize()

        screen = QGuiApplication.primaryScreen()
        g = screen.availableGeometry() if screen is not None else QRect(0, 0, 1920, 1080)
        self.move(
            g.right() - self.width() - 20,
            g.bottom() - self.height() - 20,
        )

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(0.0)

        self._fade_in = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_in.setDuration(180)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fade_out = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_out.setDuration(400)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.Type.InCubic)
        self._fade_out.finished.connect(self._on_fade_out_finished)

        self.show()
        self.raise_()
        self._fade_in.start()
        QTimer.singleShot(max(0, int(duration)), self._start_fade_out)

    def _start_fade_out(self) -> None:
        self._fade_in.stop()
        cur = max(0.0, min(1.0, float(self._opacity_effect.opacity())))
        if cur <= 0.01:
            self._on_fade_out_finished()
            return
        self._fade_out.setStartValue(cur)
        self._fade_out.setEndValue(0.0)
        self._fade_out.start()

    def _on_fade_out_finished(self) -> None:
        self.hide()
        self.deleteLater()


def show_toast(
    message: str,
    color: str = "#02C39A",
    duration: int = 3000,
    parent=None,
    title: str | None = None,
) -> None:
    Toast(message, parent=parent, color=color, duration=duration, title=title)


def toast_safe(message: str, *, duration: int = 1500, parent=None) -> None:
    """Quick green flash for safe / low-risk feedback."""
    show_toast(message, color="#43A047", duration=duration, parent=parent)
