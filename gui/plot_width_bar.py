#!/usr/bin/env python3
"""
Plot Width Bar — Compact control bar for WYSIWYG plot export sizing.

Embedded below each plot canvas. A slider controls canvas width with
snap points at common publication widths. Far-right = Auto (fill window).
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QSlider, QSpinBox, QLabel, QApplication
)
from PyQt6.QtCore import pyqtSignal, QEvent, QObject, Qt

from utils.logger_config import get_logger

logger = get_logger(__name__)

# Snap points in mm (shown as tick labels)
_SNAP_POINTS = [80, 100, 120, 140, 160, 180]
_SNAP_THRESHOLD = 3  # snap when within 3mm

# Slider range in mm — values above _MAX_MM mean "Auto"
_MIN_MM = 50
_MAX_MM = 190
_AUTO_VALUE = _MAX_MM + 10  # slider max; anything above _MAX_MM = auto


class PlotWidthBar(QWidget):
    """Compact bar with a width slider, DPI spinner, and live size label."""

    width_changed = pyqtSignal(int)   # max width in pixels, 0 = auto
    dpi_changed = pyqtSignal(int)     # export DPI

    def __init__(self, parent=None):
        super().__init__(parent)
        self._screen_dpi = self._get_screen_dpi()
        self._setup_ui()

    @staticmethod
    def _get_screen_dpi() -> float:
        screen = QApplication.primaryScreen()
        return screen.logicalDotsPerInch() if screen else 96.0

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(6)

        # Width label
        self._width_label = QLabel("Auto")
        self._width_label.setFixedWidth(52)
        self._width_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._width_label)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(_MIN_MM, _AUTO_VALUE)
        self._slider.setValue(_AUTO_VALUE)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(20)
        self._slider.setSingleStep(5)
        self._slider.setPageStep(20)
        self._slider.valueChanged.connect(self._on_slider_moved)
        self._slider.setMinimumWidth(180)
        layout.addWidget(self._slider, stretch=1)

        # Tick labels
        tick_label = QLabel("  ".join(
            [str(s) for s in _SNAP_POINTS] + ["Auto"]
        ))
        tick_label.setStyleSheet("color: #888; font-size: 9px;")
        # This is decorative — actual snap handled in code
        # We skip this label and just rely on the width_label + slider ticks

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
        # Snap to nearby preset
        for snap in _SNAP_POINTS:
            if abs(value - snap) <= _SNAP_THRESHOLD:
                if self._slider.value() != snap:
                    self._slider.blockSignals(True)
                    self._slider.setValue(snap)
                    self._slider.blockSignals(False)
                value = snap
                break

        # Auto zone
        if value > _MAX_MM:
            self._width_label.setText("Auto")
            self.width_changed.emit(0)
        else:
            self._width_label.setText(f"{value} mm")
            px = int(value * self._screen_dpi / 25.4)
            self.width_changed.emit(px)

    def update_size_label(self, width_px: int, height_px: int):
        """Update the current-size label (called on canvas resize)."""
        dpi = self._screen_dpi
        w_mm = width_px * 25.4 / dpi
        h_mm = height_px * 25.4 / dpi
        self._size_label.setText(f"Current: {w_mm:.0f} \u00d7 {h_mm:.0f} mm")

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
