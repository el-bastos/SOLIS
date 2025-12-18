#!/usr/bin/env python3
"""
Splash Screen for SOLIS Application
Shows logo and loading progress while the application initializes.
"""

from PyQt6.QtWidgets import QSplashScreen, QProgressBar, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer, QSize, QRectF, QRect
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QMovie
from pathlib import Path


class SOLISSplashScreen(QSplashScreen):
    """
    Custom splash screen for SOLIS application.

    Features:
    - SVG logo display
    - Animated loading spinner
    - Progress bar
    - Status messages
    """

    def __init__(self):
        # Load PNG logo from logo directory (1500x1500 pixels, transparent background)
        logo_path = Path(__file__).parent.parent / 'logo' / 'SOLIS-LOGO.png'

        # Create splash screen dimensions
        # Scale logo to reasonable display size: 500px wide
        splash_width = 600
        splash_height = 700  # 500 (logo) + 200 (controls area)

        pixmap = QPixmap(splash_width, splash_height)
        pixmap.fill(QColor("#F5F5F5"))  # Light gray background

        # Load and draw the PNG logo if it exists
        if logo_path.exists():
            logo_pixmap = QPixmap(str(logo_path))
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            # Scale logo to fit (500x500 from 1500x1500)
            logo_size = 500
            scaled_logo = logo_pixmap.scaled(
                logo_size, logo_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # Center logo horizontally, place at top with margin
            x = (splash_width - logo_size) // 2
            y = 40  # Top margin

            # Draw logo
            painter.drawPixmap(x, y, scaled_logo)

            # Add version/subtitle text below logo
            painter.setPen(QColor("#666666"))
            painter.setFont(QFont("Arial", 11))
            text_y = y + logo_size + 20
            painter.drawText(0, text_y, splash_width, 30,
                           Qt.AlignmentFlag.AlignCenter,
                           "Singlet Oxygen Luminescence Investigation System")

            painter.end()
        else:
            # Fallback: create simple text-based splash
            painter = QPainter(pixmap)
            painter.fillRect(pixmap.rect(), QColor("#2c3e50"))
            painter.setPen(QColor("#ecf0f1"))
            painter.setFont(QFont("Arial", 48, QFont.Weight.Bold))
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "SOLIS")
            painter.end()

        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)

        # Set window flags
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint
        )

        # Initialize progress tracking
        self.progress_value = 0
        self.status_message = "Initializing..."

        # Animation timer for spinner
        self.spinner_angle = 0
        self.spinner_timer = QTimer()
        self.spinner_timer.timeout.connect(self._update_spinner)
        self.spinner_timer.start(50)  # Update every 50ms

    def _update_spinner(self):
        """Update spinner animation."""
        self.spinner_angle = (self.spinner_angle + 10) % 360
        self.repaint()

    def showMessage(self, message: str, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                    color=QColor(Qt.GlobalColor.white)):
        """
        Override showMessage to track status.

        Parameters
        ----------
        message : str
            Status message to display
        alignment : Qt.AlignmentFlag
            Text alignment
        color : QColor
            Text color
        """
        self.status_message = message
        super().showMessage(message, alignment, color)
        self.repaint()

    def setProgress(self, value: int):
        """
        Set progress value (0-100).

        Parameters
        ----------
        value : int
            Progress percentage (0-100)
        """
        self.progress_value = min(100, max(0, value))
        self.repaint()

    def drawContents(self, painter: QPainter):
        """
        Custom drawing for splash screen.

        Draws:
        - Logo (from pixmap)
        - Spinning loader
        - Progress bar
        - Status message
        """
        # Draw the base pixmap (logo)
        painter.drawPixmap(0, 0, self.pixmap())

        # Draw spinning loader
        self._draw_spinner(painter)

        # Draw progress bar
        self._draw_progress_bar(painter)

        # Draw status message
        self._draw_status_message(painter)

    def _draw_spinner(self, painter: QPainter):
        """Draw animated spinner."""
        painter.save()

        # Spinner position (below subtitle text)
        center_x = self.width() // 2
        center_y = 610  # 40 (margin) + 500 (logo) + 20 (gap) + 30 (text) + 20 (gap)
        radius = 15

        # Draw spinner circle segments
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for i in range(12):
            angle = (self.spinner_angle + i * 30) % 360
            # Fade out older segments
            opacity = 255 - int((i / 12) * 200)
            color = QColor(255, 165, 0, opacity)  # Orange matching logo with varying opacity

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)

            # Calculate segment position
            import math
            rad = math.radians(angle)
            x = center_x + radius * math.cos(rad)
            y = center_y + radius * math.sin(rad)

            # Draw small circle for each segment
            painter.drawEllipse(int(x - 3), int(y - 3), 6, 6)

        painter.restore()

    def _draw_progress_bar(self, painter: QPainter):
        """Draw progress bar."""
        painter.save()

        # Progress bar dimensions
        bar_width = 500
        bar_height = 8
        bar_x = (self.width() - bar_width) // 2
        bar_y = 645  # Below spinner

        # Draw background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(200, 200, 200, 100))
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 4, 4)

        # Draw progress
        if self.progress_value > 0:
            progress_width = int((bar_width * self.progress_value) / 100)
            painter.setBrush(QColor(255, 165, 0))  # Orange matching logo
            painter.drawRoundedRect(bar_x, bar_y, progress_width, bar_height, 4, 4)

        painter.restore()

    def _draw_status_message(self, painter: QPainter):
        """Draw status message."""
        painter.save()

        # Message position
        msg_y = 665  # Below progress bar

        # Draw message
        painter.setPen(QColor(100, 100, 100))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(0, msg_y, self.width(), 20,
                        Qt.AlignmentFlag.AlignCenter, self.status_message)

        painter.restore()

    def finish(self, mainWin):
        """Override finish to stop animations before closing."""
        self.spinner_timer.stop()
        super().finish(mainWin)


if __name__ == '__main__':
    """Test the splash screen."""
    import sys
    from PyQt6.QtWidgets import QApplication
    import time

    app = QApplication(sys.argv)

    splash = SOLISSplashScreen()
    splash.show()

    # Simulate loading steps
    steps = [
        (10, "Loading core modules..."),
        (20, "Initializing GUI components..."),
        (40, "Loading plotters..."),
        (60, "Setting up analysis engines..."),
        (80, "Preparing data handlers..."),
        (100, "Ready!")
    ]

    for progress, message in steps:
        QTimer.singleShot(progress * 20, lambda p=progress, m=message: (
            splash.setProgress(p),
            splash.showMessage(m)
        ))

    # Close splash after 3 seconds
    QTimer.singleShot(3000, splash.close)
    QTimer.singleShot(3000, app.quit)

    sys.exit(app.exec())
