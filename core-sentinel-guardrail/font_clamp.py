"""Clamp font sizes so Qt never receives 0/negative pixel or point sizes."""


import sys

from PyQt6 import QtCore, QtGui

MIN_PX_BADGE = 7
MIN_PX_BODY = 8
MIN_PX_LABEL = 10


def clamp_font_px(size: int | float, *, floor: int = MIN_PX_BADGE) -> int:
    """Pixel sizes for QFont.setPixelSize; default floor 7 (tiny badges)."""
    if floor < 1:
        floor = 1
    return max(floor, int(round(float(size))))


def clamp_font_pt(size: int | float, *, floor: int = MIN_PX_BADGE, ceiling: int = 72) -> int:
    """Point sizes for QFont(..., pt) / setPointSize; clamp to [floor, ceiling]."""
    if ceiling < floor:
        ceiling = floor
    return max(floor, min(ceiling, int(round(float(size)))))


def normalize_application_font(app: QtGui.QGuiApplication) -> None:
    """
    Force a valid default application font.

    On Windows + Qt6 the OS default QFont often has pointSize/pixelSize -1. Copying that font
    into widgets or stylesheets can trigger:
    QFont::setPointSize: Point size <= 0 (-1), must be greater than 0
    Always set an explicit positive point size; do not clone the broken default.
    """
    nf = QtGui.QFont()
    nf.setFamily("Segoe UI")
    nf.setPointSize(10)
    app.setFont(nf)


def install_qt_message_filter() -> None:
    """Drop noisy Qt warnings about QFont::setPointSize(-1) after stylesheets / inherited fonts."""

    def _handler(
        mode: QtCore.QtMsgType,
        context: QtCore.QMessageLogContext,
        message: str,
    ) -> None:
        del mode, context
        if "QFont::setPointSize" in message and "Point size" in message:
            return
        try:
            sys.stderr.write(message + "\n")
        except OSError:
            pass

    QtCore.qInstallMessageHandler(_handler)


def paint_font_px(
    family: str,
    pixel_size: int,
    *,
    bold: bool = False,
    floor: int = MIN_PX_BADGE,
) -> QtGui.QFont:
    """
    Build a QFont for QPainter text. Do not use p.font() then setPixelSize + setBold — that
    can trigger Qt warnings: QFont::setPointSize: Point size <= 0 (-1).
    """
    f = QtGui.QFont()
    f.setFamily((family or "").strip() or "Segoe UI")
    f.setPixelSize(clamp_font_px(pixel_size, floor=floor))
    if bold:
        f.setBold(True)
    return f
