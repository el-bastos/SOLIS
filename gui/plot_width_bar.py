#!/usr/bin/env python3
"""
Plot Width Bar — Compact control bar for WYSIWYG plot export sizing.

Embedded below each plot canvas. A slider controls canvas width as a
percentage of available space. Far-right = Auto (fill window).
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QSlider, QSpinBox, QLabel, QApplication
)
from PyQt6.QtCore import pyqtSignal, QEvent, QObject, Qt

from utils.logger_config import get_logger

logger = get_logger(__name__)

# Snap points in percentage of available width
_SNAP_POINTS = [25, 40, 60, 80]
_SNAP_THRESHOLD = 3  # snap when within 3 percentage points

# Slider range — values above _MAX_PCT mean "Auto"
_MIN_PCT = 10
_MAX_PCT = 100
_AUTO_VALUE = _MAX_PCT + 10  # slider max; anything above _MAX_PCT = auto


class PlotWidthBar(QWidget):
    """Compact bar with a width slider, DPI spinner, and live size label."""

    width_changed = pyqtSignal(int)   # percentage (0 = auto, 10-100 = pct)
    dpi_changed = pyqtSignal(int)     # export DPI

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(6)

        # Width label
        self._width_label = QLabel("Auto")
        self._width_label.setFixedWidth(52)
        self._width_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self._width_label)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(_MIN_PCT, _AUTO_VALUE)
        self._slider.setValue(_AUTO_VALUE)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(10)
        self._slider.setSingleStep(5)
        self._slider.setPageStep(10)
        self._slider.valueChanged.connect(self._on_slider_moved)
        self._slider.setMinimumWidth(180)
        layout.addWidget(self._slider, stretch=1)

        layout.addSpacing(10)

        # DPI spinner
        layout.addWidget(QLabel("DPI:"))
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(72, 1200)
        self._dpi_spin.setValue(300)
        self._dpi_spin.setMaximumWidth(80)
        self._dpi_spin.valueChanged.connect(self.dpi_changed.emit)
        layout.addWidget(self._dpi_spin)

        layout.addSpacing(10)

        # Size info label
        self._size_label = QLabel("")
        self._size_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._size_label)

        layout.addStretch()

    def _on_slider_moved(self, value: int):
        """Handle slider value change with snap-to-preset logic."""
        for snap in _SNAP_POINTS:
            if abs(value - snap) <= _SNAP_THRESHOLD:
                if self._slider.value() != snap:
                    self._slider.blockSignals(True)
                    self._slider.setValue(snap)
                    self._slider.blockSignals(False)
                value = snap
                break

        if value > _MAX_PCT:
            self._width_label.setText("Auto")
            self.width_changed.emit(0)
        else:
            self._width_label.setText(f"{value}%")
            self.width_changed.emit(value)

    def update_size_label(self, width_px: int, height_px: int,
                          container_width: int = 0):
        """Update the current-size label (called on canvas resize)."""
        if container_width > 0:
            pct = width_px * 100 / container_width
            self._size_label.setText(f"Current: {pct:.0f}%")
        else:
            self._size_label.setText(f"Current: {width_px}\u00d7{height_px} px")

    def get_export_dpi(self) -> int:
        return self._dpi_spin.value()


class _CanvasResizeFilter(QObject):
    """Event filter that updates a PlotWidthBar's size label on canvas resize."""

    def __init__(self, width_bar: PlotWidthBar, parent=None):
        super().__init__(parent)
        self._width_bar = width_bar

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self._width_bar.update_size_label(
                event.size().width(), event.size().height()
            )
        return False
