"""
CoreSentinal Nudge Window - Enterprise Professional Aesthetic
Finalized for Project Integration
"""
import pyperclip
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont

class NudgeWindow(QWidget):
    COLOR_BG = QColor(25, 25, 30, 245)
    COLOR_DANGER = QColor(220, 53, 69)
    COLOR_TEXT_MAIN = QColor(245, 245, 245)
    COLOR_TEXT_SUB = QColor(180, 180, 180)
    COLOR_BORDER = QColor(70, 70, 80)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_text = ""
        self.setup_ui()
        self.setup_window_properties()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(1, 1, 1, 1)
        
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {self.COLOR_BG.name(QColor.NameFormat.HexArgb)};
                border: 1px solid {self.COLOR_BORDER.name()};
                border-radius: 4px;
            }}
            QLabel {{ color: {self.COLOR_TEXT_MAIN.name()}; border: none; background: transparent; }}
        """)
        
        layout = QVBoxLayout(self.content_widget)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 15, 15, 15)

        header_layout = QHBoxLayout()
        self.title_label = QLabel("CORE SENTINAL | COMPLIANCE")
        self.title_label.setFont(QFont("Segoe UI Variable", 9, QFont.Weight.Bold))
        self.title_label.setStyleSheet(f"color: {self.COLOR_TEXT_SUB.name()}; letter-spacing: 0.5px;")
        header_layout.addWidget(self.title_label)
        layout.addLayout(header_layout)

        self.message_label = QLabel("Sensitive Data Pattern Detected")
        self.message_label.setFont(QFont("Segoe UI Variable", 11, QFont.Weight.DemiBold))
        self.message_label.setStyleSheet(f"color: {self.COLOR_DANGER.name()};")
        layout.addWidget(self.message_label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.redact_btn = QPushButton("REDACT")
        self.synth_btn = QPushButton("SYNTHESIZE")
        
        for btn in [self.redact_btn, self.synth_btn]:
            bg = "#3A3A42" if btn == self.redact_btn else "#4A4A52"
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg}; color: white; border: 1px solid #5A5A62;
                    border-radius: 3px; font-family: 'Segoe UI'; font-size: 10px;
                    font-weight: 600; padding: 8px 15px;
                }}
                QPushButton:hover {{ background-color: #5A5A62; border-color: white; }}
            """)

        btn_layout.addWidget(self.redact_btn)
        btn_layout.addWidget(self.synth_btn)
        layout.addLayout(btn_layout)
        main_layout.addWidget(self.content_widget)

    def setup_window_properties(self):
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setFixedSize(320, 160)

    def show_nudge(self):
        screen = self.screen().availableGeometry()
        self.move(screen.width() - self.width() - 20, screen.height() - self.height() - 50)
        self.show()

    def connect_logic(self, sensitive_text):
        self.original_text = sensitive_text
        # Disconnect previous signals to avoid duplicate triggering
        try: self.redact_btn.clicked.disconnect()
        except: pass
        try: self.synth_btn.clicked.disconnect()
        except: pass
        
        self.redact_btn.clicked.connect(self.handle_redact)
        self.synth_btn.clicked.connect(self.handle_synthesize)
        self.redact_btn.setEnabled(True)
        self.synth_btn.setEnabled(True)

    def handle_redact(self):
        try:
            pyperclip.copy("[REDACTED BY CORESENTINAL]")
            self.message_label.setText("Data Successfully Redacted")
            self.message_label.setStyleSheet("color: #28a745; font-weight: bold;")
            self.redact_btn.setEnabled(False)
            self.synth_btn.setEnabled(False)
            QTimer.singleShot(1000, self.close)
        except:
            self.message_label.setText("Error updating clipboard")
            self.message_label.setStyleSheet("color: #ffc107;")

    def handle_synthesize(self):
        try:
            synthetic_data = self.generate_synthetic_replacement(self.original_text)
            pyperclip.copy(synthetic_data)
            self.message_label.setText("Data Synthesized (Faked)")
            self.message_label.setStyleSheet("color: #17a2b8; font-weight: bold;")
            self.redact_btn.setEnabled(False)
            self.synth_btn.setEnabled(False)
            QTimer.singleShot(1000, self.close)
        except:
            self.message_label.setText("Synthesis failed")
            self.message_label.setStyleSheet("color: #ffc107;")

    def generate_synthetic_replacement(self, text):
        if len(text) == 9 and text.isdigit():
            return "999001234" # Mock Rutgers RUID
        return "SYNTH-DATA-MASKED"
    def show_nudge(self):
        screen = self.screen().availableGeometry()
        self.move(screen.width() - self.width() - 20, screen.height() - self.height() - 50)
        self.show()
        
        # Start a countdown to close automatically if ignored
        # 10000 ms = 10 seconds
        QTimer.singleShot(10000, self.close_if_inactive)

    def close_if_inactive(self):
        # Only close if the user hasn't already clicked Redact or Synthesize
        if self.isVisible():
            print("[INFO] Nudge ignored by user. Closing window.")
            self.close()