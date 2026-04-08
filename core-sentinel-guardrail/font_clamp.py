"""Clamp font sizes so Qt never receives 0/negative pixel or point sizes."""

from __future__ import annotations

from PyQt6 import QtGui

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
    Ensure the application default font has a valid logical point size.

    On some Windows + Qt6 setups the default QFont reports pointSize -1 (pixel-based font only).
    Stylesheets and internal Qt code can then call setPointSize(-1) and spam:
    QFont::setPointSize: Point size <= 0 (-1), must be greater than 0
    """
    f = app.font()
    try:
        if int(f.pointSize()) > 0:
            return
    except (TypeError, ValueError):
        pass
    psf = getattr(f, "pointSizeF", None)
    if callable(psf):
        try:
            if float(psf()) > 0:
                return
        except (TypeError, ValueError):
            pass
    # Keep a concrete pt size; approximate from px when present (typical 96 dpi).
    px = 0
    try:
        px = int(f.pixelSize())
    except (TypeError, ValueError):
        px = 0
    nf = QtGui.QFont(f)
    if px > 0:
        pt = max(8, min(24, int(round(px * 72.0 / 96.0))))
        nf.setPointSize(pt)
    else:
        nf.setPointSize(10)
    app.setFont(nf)


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
