import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
# Importing your modules
from clipboard_listener import ClipboardListener
from nudge_window import NudgeWindow
from process_checker import is_already_running, print_core_sentinal_status

class CoreSentinalApp:
    def __init__(self):
        # Initialize the PyQt app (required for any UI)
        self.app = QApplication(sys.argv)
        
        # Instantiate the UI
        self.nudge = NudgeWindow()
        
        # Link the listener to the controller's processing function
        self.listener = ClipboardListener(callback=self.handle_clipboard_update)

    def handle_clipboard_update(self, text):
        if not text: return

        # Regex patterns for the demo
        is_ruid = len(text) == 9 and text.isdigit() # Rutgers ID
        is_sap = text.startswith("SAP-") and len(text) > 8 # SAP ID
        
        if is_ruid or is_sap:
            print(f"[ALERT] High-Confidence PII: {text}")
            self.nudge.connect_logic(text)
            self.nudge.show_nudge()

    def start(self):
        # Check if another instance is already running
        if is_already_running(exclude_current=True):
            print("⚠️  Another instance of CoreSentinal is already running!")
            print_core_sentinal_status()
            
            # Show warning dialog
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("CoreSentinal Already Running")
            msg.setText("Another instance of CoreSentinal is already running.")
            msg.setInformativeText("Only one instance should be active at a time.")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            
            sys.exit(1)
        
        print("CoreSentinal Service is active. Monitoring for PII...")
        print_core_sentinal_status()
        self.listener.start()
        sys.exit(self.app.exec())

if __name__ == "__main__":
    sentinal = CoreSentinalApp()
    sentinal.start()