"""Small settings dialog (Grammarly-style dark panel)."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

import user_settings
from guardrail_runtime import clear_guard_snooze, is_guard_snoozed

_BUNDLED_IDLE = Path(user_settings.__file__).resolve().parent / "assets" / "coresentinel_idle.png"

_QSS = """
QDialog { background: #1e1e1e; }
QLabel { color: #e0e0e0; font-family: 'Segoe UI'; font-size: 12px; }
QCheckBox { color: #e0e0e0; font-family: 'Segoe UI'; font-size: 12px; spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QComboBox {
  background: #2d2d2d; color: #e8e8e8; border: 1px solid #444; border-radius: 4px;
  padding: 4px 8px; min-width: 140px; font-family: 'Segoe UI'; font-size: 12px;
}
QComboBox::drop-down { border: none; width: 24px; }
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
QPushButton {
  font-family: 'Segoe UI'; font-size: 12px; padding: 6px 18px;
  border-radius: 6px; background: #3d3d3d; color: #eee; border: 1px solid #555;
}
QPushButton:hover { background: #4a4a4a; }
QPushButton#primary { background: #0d6efd; border-color: #0d6efd; color: white; }
QPushButton#primary:hover { background: #0b5ed7; }
QLineEdit {
  background: #2d2d2d; color: #e8e8e8; border: 1px solid #444; border-radius: 4px;
  padding: 6px 8px; font-family: 'Segoe UI'; font-size: 11px;
}
"""


class SentinelSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sentinel settings")
        self.setModal(True)
        self.setStyleSheet(_QSS)
        self.setMinimumWidth(360)

        d = user_settings.load()
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(20, 18, 20, 18)

        self._cb_llm = QtWidgets.QCheckBox("Monitor LLM windows only")
        self._cb_llm.setChecked(bool(d.get("monitor_llm_only", True)))
        root.addWidget(self._cb_llm)

        self._cb_badge = QtWidgets.QCheckBox("Show badge on issues")
        self._cb_badge.setChecked(bool(d.get("show_badge_on_issues", True)))
        root.addWidget(self._cb_badge)

        self._cb_block = QtWidgets.QCheckBox("Block high-risk pastes")
        self._cb_block.setChecked(bool(d.get("block_high_risk_pastes", True)))
        root.addWidget(self._cb_block)

        self._cb_sound = QtWidgets.QCheckBox("Play sound on block")
        self._cb_sound.setChecked(bool(d.get("play_sound_on_block", False)))
        root.addWidget(self._cb_sound)

        snooze_row = QtWidgets.QHBoxLayout()
        self._lbl_snooze = QtWidgets.QLabel("")
        self._lbl_snooze.setWordWrap(True)
        self._btn_clear_snooze = QtWidgets.QPushButton("Remove snooze")
        self._btn_clear_snooze.setToolTip(
            "Ends an active snooze from the remediation panel so alerts show again immediately."
        )
        self._btn_clear_snooze.clicked.connect(self._on_clear_snooze)
        snooze_row.addWidget(self._lbl_snooze, 1)
        snooze_row.addWidget(self._btn_clear_snooze, 0)
        root.addLayout(snooze_row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Sensitivity"))
        row.addStretch(1)
        self._combo_sens = QtWidgets.QComboBox()
        self._combo_sens.addItems(["Low", "Medium", "High", "Strict"])
        sens = str(d.get("sensitivity", "medium")).lower()
        idx = {"low": 0, "medium": 1, "high": 2, "strict": 3}.get(sens, 1)
        self._combo_sens.setCurrentIndex(idx)
        row.addWidget(self._combo_sens)
        root.addLayout(row)

        root.addWidget(
            QtWidgets.QLabel(
                "Idle pill icon (PNG/JPG; default: CoreSentinal logo in assets — leave empty to use it)"
            )
        )
        idle_row = QtWidgets.QHBoxLayout()
        self._edit_idle_img = QtWidgets.QLineEdit()
        self._edit_idle_img.setPlaceholderText("Custom path, or empty for bundled logo (set env none for ✓)")
        raw_idle = d.get("bubble_idle_image")
        if raw_idle and _BUNDLED_IDLE.is_file():
            try:
                if Path(str(raw_idle)).resolve() == _BUNDLED_IDLE.resolve():
                    raw_idle = ""
            except OSError:
                pass
        self._edit_idle_img.setText(str(raw_idle) if raw_idle else "")
        idle_browse = QtWidgets.QPushButton("Browse…")
        idle_browse.clicked.connect(self._browse_idle_image)
        idle_row.addWidget(self._edit_idle_img, 1)
        idle_row.addWidget(idle_browse)
        root.addLayout(idle_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)
        root.addLayout(btn_row)

    def showEvent(self, event: QtCore.QEvent) -> None:
        super().showEvent(event)
        self._refresh_snooze_ui()

    def _refresh_snooze_ui(self) -> None:
        if is_guard_snoozed():
            self._lbl_snooze.setText("Snooze is on — alerts are suppressed until it ends.")
            self._btn_clear_snooze.setEnabled(True)
        else:
            self._lbl_snooze.setText("No snooze active.")
            self._btn_clear_snooze.setEnabled(False)

    def _on_clear_snooze(self) -> None:
        clear_guard_snooze()
        self._refresh_snooze_ui()
        try:
            from toast import show_toast

            show_toast("Snooze removed — guardrail alerts active again.", color="#02C39A", duration=2500)
        except Exception:
            pass

    def _browse_idle_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose idle icon image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All files (*.*)",
        )
        if path:
            self._edit_idle_img.setText(path)

    def _on_save(self) -> None:
        sens_map = ("low", "medium", "high", "strict")
        idx = self._combo_sens.currentIndex()
        sens = sens_map[idx] if 0 <= idx < len(sens_map) else "medium"
        idle = self._edit_idle_img.text().strip()
        data = {
            "monitor_llm_only": self._cb_llm.isChecked(),
            "show_badge_on_issues": self._cb_badge.isChecked(),
            "block_high_risk_pastes": self._cb_block.isChecked(),
            "play_sound_on_block": self._cb_sound.isChecked(),
            "sensitivity": sens,
            "bubble_idle_image": idle or None,
        }
        user_settings.save(data)
        par = self.parent()
        if par is not None and hasattr(par, "reload_idle_image"):
            par.reload_idle_image()
        self.accept()
