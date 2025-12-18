#!/usr/bin/env python3
"""
Standalone splash screen launcher for SOLIS
Shows splash IMMEDIATELY, then imports and runs SOLIS
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

# Create QApplication FIRST (required for any Qt widgets)
app = QApplication(sys.argv)
app.setApplicationName("SOLIS")

# Set application icon for taskbar/dock
icon_path = Path(__file__).parent / 'logo' / 'SOLIS-ICON.png'
if icon_path.exists():
    app.setWindowIcon(QIcon(str(icon_path)))

# Import and show splash IMMEDIATELY (no heavy imports yet!)
from gui.splash_screen import SOLISSplashScreen
splash = SOLISSplashScreen()
splash.show()
splash.setProgress(0)
splash.showMessage("Loading SOLIS...")
app.processEvents()  # Force display

# NOW import the heavy solis_gui module (this takes 30 seconds)
# Splash is visible on screen while this happens
splash.setProgress(50)
splash.showMessage("Loading modules...")
app.processEvents()

import solis_gui

# Create main window
splash.setProgress(90)
splash.showMessage("Initializing...")
app.processEvents()

window = solis_gui.SOLISMainWindow()

# Done
splash.setProgress(100)
splash.showMessage("Ready!")
app.processEvents()

# Show window and close splash
window.show()
splash.finish(window)

sys.exit(app.exec())
