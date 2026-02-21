#!/usr/bin/env python3
"""
Plot Appearance Dialog — Artist-based editor for SOLIS plots.

Opened via right-click on any displayed plot. Inspects the figure's
actual matplotlib artists (lines, scatter, text, grid) and generates
controls dynamically. Apply modifies artists in-place — no replot needed.

Sections: Color Palette, Lines & Markers, Fonts, Grid.
"""

import matplotlib
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QPushButton, QHBoxLayout, QGroupBox, QLabel, QSpinBox,
    QScrollArea, QWidget, QComboBox, QCheckBox, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFontDatabase
from gui.stylesheets import INFO_LABEL_STYLE
from utils.logger_config import get_logger

logger = get_logger(__name__)

# ── Linestyle codes ──────────────────────────────────────────────────

_LS_ITEMS = [('Solid', '-'), ('Dashed', '--'), ('Dotted', ':'), ('Dash-dot', '-.')]
_LS_TO_IDX = {code: i for i, (_, code) in enumerate(_LS_ITEMS)}

_GRID_ITEMS = [('Dashed', '--'), ('Dotted', ':'), ('Solid', '-')]
_GRID_TO_IDX = {code: i for i, (_, code) in enumerate(_GRID_ITEMS)}

# ── Color palettes ───────────────────────────────────────────────────

COLOR_PALETTES = {
    'SOLIS Default':  ['#646464', '#CC0000', '#FF8C00', '#228B22', '#4169E1', '#8B008B', '#DC143C', '#FFD700'],
    'Grayscale':      ['#333333', '#666666', '#999999', '#BBBBBB', '#444444', '#777777', '#555555', '#AAAAAA'],
    'Colorblind':     ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],
    'Nature':         ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000'],
}

_PALETTE_NAMES = ['Custom (no change)'] + list(COLOR_PALETTES.keys())


def _to_hex(color) -> str:
    """Normalize any matplotlib color to hex string."""
    try:
        return mcolors.to_hex(color, keep_alpha=False)
    except (ValueError, TypeError):
        return '#000000'


def _normalize_linestyle(ls) -> str:
    """Normalize linestyle to a string code ('-', '--', ':', '-.')."""
    if isinstance(ls, str):
        _name_map = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdot': '-.'}
        return _name_map.get(ls, ls)
    if isinstance(ls, tuple) and len(ls) == 2:
        _offset, seq = ls
        if seq is None or seq == () or (isinstance(seq, (list, tuple)) and len(seq) == 0):
            return '-'
        if isinstance(seq, (list, tuple)):
            if len(seq) == 2:
                on, off = seq
                return ':' if on < off else '--'
            elif len(seq) >= 4:
                return '-.'
        return '--'
    return '-'


class PlotAppearanceDialog(QDialog):
    """Right-click dialog that inspects and edits a figure's artists directly."""

    def __init__(self, figure, canvas, parent=None):
        super().__init__(parent)
        self._fig = figure
        self._canvas = canvas

        self.setWindowTitle("Plot Appearance")
        self.setModal(True)

        self._line_entries = []
        self._scatter_entries = []
        self._discover_and_snapshot()
        self._setup_ui()

    # -----------------------------------------------------------------
    # Discovery
    # -----------------------------------------------------------------

    def _discover_and_snapshot(self):
        """Walk figure artists and record current values for Reset."""
        fig = self._fig

        # --- Lines (Line2D) ---
        lines = []
        for ax in fig.get_axes():
            for line in ax.get_lines():
                label = line.get_label()
                if not label or label.startswith('_'):
                    continue
                has_marker = (line.get_marker() is not None
                              and str(line.get_marker()) not in ('None', '', ' '))
                lines.append({
                    'artist': line,
                    'label': label,
                    'color': _to_hex(line.get_color()),
                    'linewidth': line.get_linewidth(),
                    'linestyle': _normalize_linestyle(line.get_linestyle()),
                    'marker': line.get_marker(),
                    'markersize': line.get_markersize(),
                    'has_marker': has_marker,
                })
        self._discovered_lines = lines

        # --- Scatter (PathCollection from ax.scatter()) ---
        scatters = []
        for ax in fig.get_axes():
            for coll in ax.collections:
                if not isinstance(coll, PathCollection):
                    continue
                label = coll.get_label()
                if not label or label.startswith('_'):
                    continue
                fc = coll.get_facecolors()
                color = _to_hex(fc[0] if len(fc) > 0 else '#000000')
                sizes = coll.get_sizes()
                size = float(sizes[0]) if len(sizes) > 0 else 20.0
                alpha = coll.get_alpha()
                if alpha is None:
                    alpha = float(fc[0][3]) if len(fc) > 0 and len(fc[0]) >= 4 else 1.0
                scatters.append({
                    'artist': coll,
                    'label': label,
                    'color': color,
                    'size': size,
                    'alpha': float(alpha),
                })
        self._discovered_scatters = scatters

        # --- Fonts ---
        title_size = 14.0
        title_weight = 'bold'
        if fig.texts:
            title_size = fig.texts[0].get_fontsize() or 14.0
            title_weight = fig.texts[0].get_fontweight() or 'bold'
        elif fig.get_axes():
            t = fig.get_axes()[0].get_title()
            if t:
                title_size = fig.get_axes()[0].title.get_fontsize() or 14.0
                title_weight = fig.get_axes()[0].title.get_fontweight() or 'bold'

        axis_size = 12.0
        for ax in fig.get_axes():
            fs = ax.xaxis.label.get_fontsize()
            if fs:
                axis_size = fs
                break

        tick_size = 10.0
        for ax in fig.get_axes():
            tl = ax.get_xticklabels()
            if tl:
                ts = tl[0].get_fontsize()
                if ts:
                    tick_size = ts
                break

        legend_size = 10.0
        for ax in fig.get_axes():
            leg = ax.get_legend()
            if leg and leg.get_texts():
                legend_size = leg.get_texts()[0].get_fontsize() or 10.0
                break

        font_family = 'Arial'
        all_text = fig.findobj(matplotlib.text.Text)
        for t in all_text:
            fam = t.get_fontfamily()
            if fam:
                font_family = fam[0] if isinstance(fam, list) else fam
                break

        self._snap_fonts = {
            'title_size': float(title_size),
            'title_weight': str(title_weight),
            'axis_size': float(axis_size),
            'tick_size': float(tick_size),
            'legend_size': float(legend_size),
            'font_family': font_family,
        }

        # --- Grid ---
        grid_on = False
        grid_alpha = 0.3
        grid_ls = '--'
        for ax in fig.get_axes():
            gridlines = ax.xaxis.get_gridlines()
            if gridlines:
                grid_on = gridlines[0].get_visible()
                grid_alpha = gridlines[0].get_alpha() or 0.3
                grid_ls = _normalize_linestyle(gridlines[0].get_linestyle())
                break

        self._snap_grid = {
            'on': grid_on,
            'alpha': grid_alpha,
            'linestyle': grid_ls,
        }

        # Deep-copy snapshots (without artist references)
        self._snap_lines = [
            {k: v for k, v in entry.items() if k != 'artist'}
            for entry in lines
        ]
        self._snap_scatters = [
            {k: v for k, v in entry.items() if k != 'artist'}
            for entry in scatters
        ]

    # -----------------------------------------------------------------
    # UI Setup
    # -----------------------------------------------------------------

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        self._setup_palette_section(layout)
        self._setup_lines_section(layout)
        self._setup_fonts_section(layout)
        self._setup_grid_section(layout)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        # Buttons
        btn_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Restore all values to when dialog was opened")
        reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(reset_btn)

        btn_layout.addStretch()

        apply_btn = QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)

        main_layout.addLayout(btn_layout)

        self.setMinimumWidth(460)
        self.setMinimumHeight(480)

    def _setup_palette_section(self, parent_layout):
        group = QGroupBox("Color Palette")
        layout = QVBoxLayout()

        note = QLabel("Apply a predefined color palette to all series.")
        note.setStyleSheet(INFO_LABEL_STYLE)
        note.setWordWrap(True)
        layout.addWidget(note)

        # Palette combo
        row = QHBoxLayout()
        row.addWidget(QLabel("Palette:"))
        self._palette_combo = QComboBox()
        self._palette_combo.addItems(_PALETTE_NAMES)
        self._palette_combo.setCurrentIndex(0)
        self._palette_combo.currentIndexChanged.connect(self._on_palette_preview)
        row.addWidget(self._palette_combo, stretch=1)
        layout.addLayout(row)

        # Preview squares
        self._palette_preview_widgets = []
        preview_row = QHBoxLayout()
        preview_row.setSpacing(3)
        for _ in range(8):
            swatch = QLabel()
            swatch.setFixedSize(24, 18)
            swatch.setStyleSheet("background-color: #444; border: 1px solid #666; border-radius: 2px;")
            self._palette_preview_widgets.append(swatch)
            preview_row.addWidget(swatch)
        preview_row.addStretch()
        layout.addLayout(preview_row)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _on_palette_preview(self, index: int):
        """Update preview swatches when palette selection changes."""
        name = self._palette_combo.currentText()
        colors = COLOR_PALETTES.get(name, None)
        for i, swatch in enumerate(self._palette_preview_widgets):
            if colors and i < len(colors):
                swatch.setStyleSheet(
                    f"background-color: {colors[i]}; border: 1px solid #666; border-radius: 2px;"
                )
            else:
                swatch.setStyleSheet(
                    "background-color: #444; border: 1px solid #666; border-radius: 2px;"
                )

    def _setup_lines_section(self, parent_layout):
        n_total = len(self._discovered_lines) + len(self._discovered_scatters)
        if n_total == 0:
            return

        group = QGroupBox(f"Lines & Markers ({n_total})")
        layout = QVBoxLayout()

        # --- Line2D entries ---
        self._line_entries = []
        for info in self._discovered_lines:
            entry = {'artist': info['artist']}
            row = QHBoxLayout()

            # Label
            lbl = QLabel(info['label'])
            lbl.setMinimumWidth(160)
            row.addWidget(lbl, stretch=1)

            # Line width
            lw_spin = QDoubleSpinBox()
            lw_spin.setRange(0.5, 8.0)
            lw_spin.setSingleStep(0.5)
            lw_spin.setDecimals(1)
            lw_spin.setValue(info['linewidth'])
            lw_spin.setPrefix("lw:")
            lw_spin.setMaximumWidth(75)
            entry['lw_spin'] = lw_spin
            row.addWidget(lw_spin)

            if info['has_marker']:
                ms_spin = QDoubleSpinBox()
                ms_spin.setRange(0.5, 15.0)
                ms_spin.setSingleStep(0.5)
                ms_spin.setDecimals(1)
                ms_spin.setValue(info['markersize'])
                ms_spin.setPrefix("ms:")
                ms_spin.setMaximumWidth(75)
                entry['ms_spin'] = ms_spin
                row.addWidget(ms_spin)
            else:
                ls_combo = QComboBox()
                for name, code in _LS_ITEMS:
                    ls_combo.addItem(name, code)
                idx = _LS_TO_IDX.get(info['linestyle'], 0)
                ls_combo.setCurrentIndex(idx)
                ls_combo.setMaximumWidth(90)
                entry['ls_combo'] = ls_combo
                row.addWidget(ls_combo)

            self._line_entries.append(entry)
            layout.addLayout(row)

        # --- Scatter (PathCollection) entries ---
        self._scatter_entries = []
        for info in self._discovered_scatters:
            entry = {'artist': info['artist']}
            row = QHBoxLayout()

            # Label
            lbl = QLabel(info['label'])
            lbl.setMinimumWidth(160)
            row.addWidget(lbl, stretch=1)

            # Scatter size
            sz_spin = QDoubleSpinBox()
            sz_spin.setRange(1.0, 200.0)
            sz_spin.setSingleStep(5.0)
            sz_spin.setDecimals(0)
            sz_spin.setValue(info['size'])
            sz_spin.setPrefix("s:")
            sz_spin.setMaximumWidth(75)
            entry['sz_spin'] = sz_spin
            row.addWidget(sz_spin)

            # Scatter alpha
            alpha_spin = QDoubleSpinBox()
            alpha_spin.setRange(0.05, 1.0)
            alpha_spin.setSingleStep(0.1)
            alpha_spin.setDecimals(2)
            alpha_spin.setValue(info['alpha'])
            alpha_spin.setPrefix("\u03b1:")
            alpha_spin.setMaximumWidth(75)
            entry['alpha_spin'] = alpha_spin
            row.addWidget(alpha_spin)

            self._scatter_entries.append(entry)
            layout.addLayout(row)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _setup_fonts_section(self, parent_layout):
        group = QGroupBox("Fonts")
        form = QFormLayout()
        snap = self._snap_fonts

        self._font_family_combo = QComboBox()
        families = QFontDatabase.families()
        self._font_family_combo.addItems(families)
        idx = self._font_family_combo.findText(snap['font_family'])
        if idx >= 0:
            self._font_family_combo.setCurrentIndex(idx)
        form.addRow("Family:", self._font_family_combo)

        self._title_size_spin = QSpinBox()
        self._title_size_spin.setRange(6, 36)
        self._title_size_spin.setValue(int(snap['title_size']))
        self._title_size_spin.setSuffix(" pt")
        form.addRow("Title:", self._title_size_spin)

        self._title_weight_combo = QComboBox()
        self._title_weight_combo.addItems(["Bold", "Normal"])
        if snap['title_weight'] in ('normal', '400'):
            self._title_weight_combo.setCurrentIndex(1)
        form.addRow("Title weight:", self._title_weight_combo)

        self._axis_size_spin = QSpinBox()
        self._axis_size_spin.setRange(6, 36)
        self._axis_size_spin.setValue(int(snap['axis_size']))
        self._axis_size_spin.setSuffix(" pt")
        form.addRow("Axis labels:", self._axis_size_spin)

        self._tick_size_spin = QSpinBox()
        self._tick_size_spin.setRange(6, 36)
        self._tick_size_spin.setValue(int(snap['tick_size']))
        self._tick_size_spin.setSuffix(" pt")
        form.addRow("Tick labels:", self._tick_size_spin)

        self._legend_size_spin = QSpinBox()
        self._legend_size_spin.setRange(6, 36)
        self._legend_size_spin.setValue(int(snap['legend_size']))
        self._legend_size_spin.setSuffix(" pt")
        form.addRow("Legend:", self._legend_size_spin)

        group.setLayout(form)
        parent_layout.addWidget(group)

    def _setup_grid_section(self, parent_layout):
        group = QGroupBox("Grid")
        form = QFormLayout()
        snap = self._snap_grid

        self._grid_check = QCheckBox("Show grid lines")
        self._grid_check.setChecked(snap['on'])
        form.addRow(self._grid_check)

        self._grid_style_combo = QComboBox()
        for name, code in _GRID_ITEMS:
            self._grid_style_combo.addItem(name, code)
        idx = _GRID_TO_IDX.get(snap['linestyle'], 0)
        self._grid_style_combo.setCurrentIndex(idx)
        form.addRow("Style:", self._grid_style_combo)

        self._grid_alpha_spin = QDoubleSpinBox()
        self._grid_alpha_spin.setRange(0.0, 1.0)
        self._grid_alpha_spin.setSingleStep(0.1)
        self._grid_alpha_spin.setDecimals(1)
        self._grid_alpha_spin.setValue(snap['alpha'])
        form.addRow("Opacity:", self._grid_alpha_spin)

        group.setLayout(form)
        parent_layout.addWidget(group)

    # -----------------------------------------------------------------
    # Apply
    # -----------------------------------------------------------------

    def _on_apply(self):
        """Modify figure artists in-place and redraw canvas."""
        fig = self._fig

        # --- Color palette ---
        palette_name = self._palette_combo.currentText()
        palette_colors = COLOR_PALETTES.get(palette_name, None)

        if palette_colors:
            # Collect all discovered artists in order
            all_artists = []
            for entry in self._line_entries:
                all_artists.append(('line', entry['artist']))
            for entry in self._scatter_entries:
                all_artists.append(('scatter', entry['artist']))

            for i, (atype, artist) in enumerate(all_artists):
                hex_c = palette_colors[i % len(palette_colors)]
                if atype == 'line':
                    artist.set_color(hex_c)
                    artist.set_markerfacecolor(hex_c)
                    artist.set_markeredgecolor(hex_c)
                else:
                    # PathCollection
                    alpha = artist.get_alpha()
                    if alpha is None:
                        fc = artist.get_facecolors()
                        alpha = float(fc[0][3]) if len(fc) > 0 and len(fc[0]) >= 4 else 1.0
                    rgba = mcolors.to_rgba(hex_c, alpha=alpha)
                    artist.set_facecolors([rgba])
                    artist.set_edgecolors([rgba])

        # --- Lines (Line2D) — line width, marker size, linestyle ---
        for entry in self._line_entries:
            line = entry['artist']
            line.set_linewidth(entry['lw_spin'].value())
            if 'ms_spin' in entry:
                line.set_markersize(entry['ms_spin'].value())
            if 'ls_combo' in entry:
                line.set_linestyle(entry['ls_combo'].currentData())

        # --- Scatter (PathCollection) — size and alpha ---
        for entry in self._scatter_entries:
            coll = entry['artist']
            coll.set_sizes([entry['sz_spin'].value()])
            # Update alpha if changed (preserve current color)
            alpha = entry['alpha_spin'].value()
            fc = coll.get_facecolors()
            if len(fc) > 0:
                new_rgba = list(fc[0])
                new_rgba[3] = alpha
                coll.set_facecolors([new_rgba])
                new_rgba_edge = list(coll.get_edgecolors()[0]) if len(coll.get_edgecolors()) > 0 else new_rgba[:]
                new_rgba_edge[3] = alpha
                coll.set_edgecolors([new_rgba_edge])

        # --- Recreate legends ---
        self._refresh_legends()

        # --- Fonts ---
        family = self._font_family_combo.currentText()
        title_size = self._title_size_spin.value()
        title_weight = 'bold' if self._title_weight_combo.currentIndex() == 0 else 'normal'
        axis_size = self._axis_size_spin.value()
        tick_size = self._tick_size_spin.value()
        legend_size = self._legend_size_spin.value()

        for item in fig.findobj(matplotlib.text.Text):
            item.set_fontfamily(family)

        for text in fig.texts:
            text.set_fontsize(title_size)
            text.set_fontweight(title_weight)

        for ax in fig.get_axes():
            if ax.get_title():
                ax.title.set_fontsize(title_size)
                ax.title.set_fontweight(title_weight)
            ax.xaxis.label.set_fontsize(axis_size)
            ax.yaxis.label.set_fontsize(axis_size)
            # Standard 90° y-axis labels
            if ax.yaxis.label.get_text():
                ax.yaxis.label.set_rotation(90)
                ax.yaxis.label.set_ha('center')
                ax.yaxis.label.set_va('bottom')
            ax.tick_params(labelsize=tick_size)
            leg = ax.get_legend()
            if leg:
                for t in leg.get_texts():
                    t.set_fontsize(legend_size)

        # Align y-labels across subplots
        if len(fig.get_axes()) > 1:
            fig.align_ylabels(fig.get_axes())

        # --- Grid (explicit off then on to avoid toggle ambiguity) ---
        grid_on = self._grid_check.isChecked()
        grid_ls = self._grid_style_combo.currentData()
        grid_alpha = self._grid_alpha_spin.value()
        for ax in fig.get_axes():
            ax.grid(False, which='both')
            if grid_on:
                ax.grid(True, which='major', linestyle=grid_ls, alpha=grid_alpha)

        # Synchronous redraw
        self._canvas.draw()
        QApplication.processEvents()
        logger.info("Plot appearance applied")

    # -----------------------------------------------------------------
    # Legend refresh
    # -----------------------------------------------------------------

    def _refresh_legends(self):
        """Recreate legends on all axes so swatches match modified artists."""
        for ax in self._fig.get_axes():
            old_leg = ax.get_legend()
            if old_leg is None:
                continue
            fontsize = None
            texts = old_leg.get_texts()
            if texts:
                fontsize = texts[0].get_fontsize()
            loc = getattr(old_leg, '_loc', 'best')
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                old_leg.remove()
                ax.legend(handles=handles, labels=labels,
                          loc=loc, fontsize=fontsize)

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------

    def _on_reset(self):
        """Restore all artist properties and controls to dialog-open state."""
        fig = self._fig

        # --- Lines ---
        for entry, snap in zip(self._line_entries, self._snap_lines):
            line = entry['artist']
            line.set_color(snap['color'])
            line.set_markerfacecolor(snap['color'])
            line.set_markeredgecolor(snap['color'])
            line.set_linewidth(snap['linewidth'])
            line.set_linestyle(snap['linestyle'])
            line.set_markersize(snap['markersize'])

            entry['lw_spin'].setValue(snap['linewidth'])
            if 'ms_spin' in entry:
                entry['ms_spin'].setValue(snap['markersize'])
            if 'ls_combo' in entry:
                idx = _LS_TO_IDX.get(snap['linestyle'], 0)
                entry['ls_combo'].setCurrentIndex(idx)

        # --- Scatter ---
        for entry, snap in zip(self._scatter_entries, self._snap_scatters):
            coll = entry['artist']
            rgba = mcolors.to_rgba(snap['color'], alpha=snap['alpha'])
            coll.set_facecolors([rgba])
            coll.set_edgecolors([rgba])
            coll.set_sizes([snap['size']])

            entry['sz_spin'].setValue(snap['size'])
            entry['alpha_spin'].setValue(snap['alpha'])

        # --- Recreate legends ---
        self._refresh_legends()

        # --- Palette combo ---
        self._palette_combo.setCurrentIndex(0)

        # --- Fonts ---
        sf = self._snap_fonts
        idx = self._font_family_combo.findText(sf['font_family'])
        if idx >= 0:
            self._font_family_combo.setCurrentIndex(idx)
        self._title_size_spin.setValue(int(sf['title_size']))
        self._title_weight_combo.setCurrentIndex(
            0 if sf['title_weight'] not in ('normal', '400') else 1
        )
        self._axis_size_spin.setValue(int(sf['axis_size']))
        self._tick_size_spin.setValue(int(sf['tick_size']))
        self._legend_size_spin.setValue(int(sf['legend_size']))

        for item in fig.findobj(matplotlib.text.Text):
            item.set_fontfamily(sf['font_family'])
        for text in fig.texts:
            text.set_fontsize(sf['title_size'])
            text.set_fontweight(sf['title_weight'])
        for ax in fig.get_axes():
            if ax.get_title():
                ax.title.set_fontsize(sf['title_size'])
                ax.title.set_fontweight(sf['title_weight'])
            ax.xaxis.label.set_fontsize(sf['axis_size'])
            ax.yaxis.label.set_fontsize(sf['axis_size'])
            # Standard 90° y-axis labels
            if ax.yaxis.label.get_text():
                ax.yaxis.label.set_rotation(90)
                ax.yaxis.label.set_ha('center')
                ax.yaxis.label.set_va('bottom')
            ax.tick_params(labelsize=sf['tick_size'])
            leg = ax.get_legend()
            if leg:
                for t in leg.get_texts():
                    t.set_fontsize(sf['legend_size'])

        # Align y-labels across subplots
        if len(fig.get_axes()) > 1:
            fig.align_ylabels(fig.get_axes())

        # --- Grid (explicit off then on) ---
        sg = self._snap_grid
        self._grid_check.setChecked(sg['on'])
        idx = _GRID_TO_IDX.get(sg['linestyle'], 0)
        self._grid_style_combo.setCurrentIndex(idx)
        self._grid_alpha_spin.setValue(sg['alpha'])
        for ax in fig.get_axes():
            ax.grid(False, which='both')
            if sg['on']:
                ax.grid(True, which='major', linestyle=sg['linestyle'], alpha=sg['alpha'])

        self._canvas.draw()
        QApplication.processEvents()
        logger.info("Plot appearance reset to original values")
