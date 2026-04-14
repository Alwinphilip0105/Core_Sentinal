"""
Vertical slide-out toolbar for RiskBubble (40px wide, icon buttons).
"""

from __future__ import annotations

import html
from typing import Any, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from font_clamp import MIN_PX_LABEL, paint_font_px


class ToolbarTooltip(QtWidgets.QWidget):
    """Floating label next to the toolbar (no native QToolTip — unreliable on frameless Tool windows)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(
            parent,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._label = QtWidgets.QLabel(self)
        self._label.setStyleSheet(
            """
            background-color: rgba(28, 28, 30, 245);
            color: white;
            font-size: 12px;
            font-weight: 500;
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.12);
            """
        )
        self._label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._label.adjustSize()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self.adjustSize()
        self.hide()

        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(150)
        self._hide_timer.timeout.connect(self.hide)

    def schedule_hide(self) -> None:
        self._hide_timer.start(150)

    def show_tooltip(
        self,
        main_text: str,
        shortcut: str,
        near_widget: QtWidgets.QWidget,
        toolbar_x: int,
    ) -> None:
        self._hide_timer.stop()
        self._label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        em = html.escape(main_text)
        es = html.escape(shortcut)
        self._label.setText(
            f"<b>{em}</b><br><span style=\"color:#888;font-size:10px;\">{es}</span>"
        )
        self._label.adjustSize()
        self.adjustSize()

        screen = QtGui.QGuiApplication.primaryScreen()
        g = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1920, 1080)
        gp = near_widget.mapToGlobal(QtCore.QPoint(0, 0))
        wy = gp.y() + near_widget.height() // 2 - self.height() // 2

        if toolbar_x > g.left() + g.width() // 2:
            wx = toolbar_x - self.width() - 8
        else:
            wx = toolbar_x + 60

        wy = max(g.top(), min(wy, g.bottom() - self.height()))
        wx = max(g.left(), min(wx, g.right() - self.width()))
        self.move(wx, wy)
        self.show()
        self.raise_()


class ToolbarIconButton(QtWidgets.QWidget):
    """40×44 icon button with hover/press paint states."""

    clicked = QtCore.pyqtSignal()

    def __init__(
        self,
        icon: str,
        color: QtGui.QColor,
        bubble: Any,
        tooltip_main: str,
        shortcut_hint: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._icon = icon
        self._color = color
        self._bubble = bubble
        self._tooltip_main = tooltip_main
        self._shortcut_hint = shortcut_hint
        self.setFixedSize(40, 44)
        self.setMouseTracking(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._hover = False
        self._pressed = False

    def set_icon_color(self, color: QtGui.QColor) -> None:
        self._color = color
        self.update()

    def enterEvent(self, event: QtCore.QEvent) -> None:
        self._hover = True
        self.update()
        b = self._bubble
        if b is not None and getattr(b, "_tb_tooltip", None) is not None:
            tb = self.parent()
            toolbar_x = (
                tb.mapToGlobal(QtCore.QPoint(0, 0)).x()
                if isinstance(tb, QtWidgets.QWidget)
                else 0
            )
            b._tb_tooltip.show_tooltip(self._tooltip_main, self._shortcut_hint, self, toolbar_x)
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover = False
        self._pressed = False
        self.update()
        b = self._bubble
        if b is not None and getattr(b, "_tb_tooltip", None) is not None:
            b._tb_tooltip.schedule_hide()
        if b is not None:
            QtCore.QTimer.singleShot(200, b._maybe_hide_toolbar)
        super().leaveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pressed = True
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pressed = False
            self.update()
            if self.rect().contains(event.position().toPoint()):
                self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = self.rect()
        if self._pressed:
            p.fillRect(r, QtGui.QColor(255, 255, 255, 38))
        elif self._hover:
            p.fillRect(r, QtGui.QColor(255, 255, 255, 20))
        op = 1.0 if (self._hover or self._pressed) else 0.4
        p.setOpacity(op)
        p.setPen(self._color)
        fam = p.font().family() or "Segoe UI"
        p.setFont(paint_font_px(fam, 14, floor=MIN_PX_LABEL))
        p.drawText(r, QtCore.Qt.AlignmentFlag.AlignCenter, self._icon)


class BubbleToolbar(QtWidgets.QWidget):
    """Vertical icon strip docked inside RiskBubble (same top-level window as the pill).

    Monitoring on/off is only on the main pill (and Alt+G) — no duplicate power row here.
    """

    rephrase_clicked = QtCore.pyqtSignal()
    redact_clicked = QtCore.pyqtSignal()
    encrypt_clicked = QtCore.pyqtSignal()
    scan_file_clicked = QtCore.pyqtSignal()
    settings_clicked = QtCore.pyqtSignal()

    def __init__(self, bubble: Any, parent: Optional[QtWidgets.QWidget] = None):
        # Child of RiskBubble so icons share one window with the pill (no extra Tool window).
        super().__init__(bubble)
        self._bubble = bubble
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName("bubbleToolbar")
        self.setStyleSheet(
            "#bubbleToolbar { background-color: rgba(28,28,30,235); "
            "border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); }"
        )

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(0)

        white = QtGui.QColor("#ffffff")
        amber = QtGui.QColor("#FFD740")
        blue = QtGui.QColor("#42A5F5")

        self._btn_rephrase = ToolbarIconButton(
            "✏",
            white,
            self._bubble,
            "Rewrite text without PII",
            "[Alt+R]",
        )
        self._btn_rephrase.clicked.connect(self.rephrase_clicked.emit)
        lay.addWidget(self._btn_rephrase)

        self._btn_redact = ToolbarIconButton(
            "▓",
            amber,
            self._bubble,
            "Mask PII (partial digits, masked email/IP, asterisks elsewhere)",
            "[Alt+D]",
        )
        self._btn_redact.clicked.connect(self.redact_clicked.emit)
        lay.addWidget(self._btn_redact)

        self._btn_encrypt = ToolbarIconButton(
            "🔒",
            blue,
            self._bubble,
            "Encrypt sensitive content",
            "[Alt+E]",
        )
        self._btn_encrypt.clicked.connect(self.encrypt_clicked.emit)
        lay.addWidget(self._btn_encrypt)

        self._btn_scan = ToolbarIconButton(
            "⬆",
            white,
            self._bubble,
            "Upload/scan document for PII (Gemini)",
            "[Alt+S]",
        )
        self._btn_scan.clicked.connect(self.scan_file_clicked.emit)
        lay.addWidget(self._btn_scan)

        self._btn_settings = ToolbarIconButton(
            "⚙",
            white,
            self._bubble,
            "Open settings",
            "[Alt+,]",
        )
        self._btn_settings.clicked.connect(self.settings_clicked.emit)
        lay.addWidget(self._btn_settings)

        self.setFixedWidth(40)
        self.adjustSize()

    def enterEvent(self, event: QtCore.QEvent) -> None:
        b = self._bubble
        if b is not None and hasattr(b, "_toolbar_content_enter"):
            b._toolbar_content_enter()
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        b = self._bubble
        if b is not None and getattr(b, "_tb_tooltip", None) is not None:
            b._tb_tooltip.schedule_hide()
        if b is not None:
            QtCore.QTimer.singleShot(200, b._maybe_hide_toolbar)
        super().leaveEvent(event)
