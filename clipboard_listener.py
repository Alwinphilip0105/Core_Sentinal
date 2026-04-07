import win32gui
import win32con
import win32clipboard
import ctypes  # We use this to talk directly to Windows DLLs
from typing import Callable, Optional
from PyQt6.QtCore import QThread, pyqtSignal
import time

# --- FAIL-SAFE DEFINITIONS ---
# If pywin32 is missing these, we define them manually
WM_CLIPBOARDUPDATE = 0x031D 

def add_clipboard_listener(hwnd):
    # This reaches directly into the Windows User32 library
    return ctypes.windll.user32.AddClipboardFormatListener(hwnd)

def remove_clipboard_listener(hwnd):
    return ctypes.windll.user32.RemoveClipboardFormatListener(hwnd)
# -----------------------------

class ClipboardListenerThread(QThread):
    clipboard_updated = pyqtSignal(str)
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        super().__init__()
        self.callback = callback
        self.running = False
        self.hwnd = None

    def run(self):
        try:
            wc = win32gui.WNDCLASS()
            self.hinst = wc.hInstance = win32gui.GetModuleHandle(None)
            wc.lpszClassName = "CoreSentinalMonitor"
            wc.lpfnWndProc = self._wnd_proc
            
            class_atom = win32gui.RegisterClass(wc)
            self.hwnd = win32gui.CreateWindowEx(
                0, class_atom, None, 0, 0, 0, 0, 0,
                win32con.HWND_MESSAGE, 0, self.hinst, None
            )
            
            # Use our new manual listener function here
            add_clipboard_listener(self.hwnd)
            
            self.running = True
            while self.running:
                # Use a small timeout so we don't freeze the thread
                win32gui.PumpWaitingMessages()
                time.sleep(0.01)
                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        # IMPORTANT: Use 'WM_CLIPBOARDUPDATE' (our manual variable) 
        # NOT 'win32con.WM_CLIPBOARDUPDATE'
        if msg == WM_CLIPBOARDUPDATE:
            self.process_clipboard()
            
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def process_clipboard(self):
        try:
            win32clipboard.OpenClipboard()
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                self.clipboard_updated.emit(data)
                if self.callback:
                    self.callback(data)
            win32clipboard.CloseClipboard()
        except Exception as e:
            print(f"Clipboard access error: {e}")
    def stop(self):
        """Safely stops the thread and removes the Windows hook."""
        self.running = False
        if self.hwnd:
            remove_clipboard_listener(self.hwnd)
        self.quit()
        self.wait()

    def cleanup(self):
        """Releases Windows resources."""
        if self.hwnd:
            try:
                win32gui.DestroyWindow(self.hwnd)
            except:
                pass

class ClipboardListener:
    """
    Main interface class that main.py imports.
    It manages the background thread to keep the UI responsive.
    """
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        self.callback = callback
        self.thread = None
        
    def start(self):
        """Starts monitoring clipboard changes in a separate thread."""
        if self.thread and self.thread.isRunning():
            return
        
        self.thread = ClipboardListenerThread(callback=self.callback)
        self.thread.start()
    
    def stop(self):
        """Stops the listener and cleans up resources."""
        if self.thread:
            self.thread.stop()
            self.thread = None

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.isRunning()