#!/usr/bin/env python3
"""
Origin Exporter — Export matplotlib figures to OriginLab .opju projects.

Walks the figure's actual artists (Line2D, PathCollection) to extract
plotted data and styling, then recreates the plot in Origin with matching
worksheets and graphs. Requires Origin installed + originpro package.
"""

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection

from utils.logger_config import get_logger

logger = get_logger(__name__)

# ── Style mapping tables ─────────────────────────────────────────────

_LS_MAP = {'-': 0, 'solid': 0, '--': 1, 'dashed': 1,
           ':': 2, 'dotted': 2, '-.': 3, 'dashdot': 3}

_MARKER_MAP = {'o': 1, '.': 1, 's': 2, '^': 3, 'v': 4,
               'd': 5, 'D': 5, '+': 6, 'x': 7, '*': 8}


# ── Helpers ───────────────────────────────────────────────────────────

def _to_hex(color) -> str:
    try:
        return mcolors.to_hex(color, keep_alpha=False)
    except (ValueError, TypeError):
        return '#000000'


def _normalize_ls(ls) -> str:
    """Normalize matplotlib linestyle (string or tuple) to a string code."""
    if isinstance(ls, str):
        name_map = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdot': '-.'}
        return name_map.get(ls, ls)
    if isinstance(ls, tuple) and len(ls) == 2:
        _, seq = ls
        if seq is None or seq == () or (isinstance(seq, (list, tuple)) and len(seq) == 0):
            return '-'
        if isinstance(seq, (list, tuple)):
            if len(seq) == 2:
                return ':' if seq[0] < seq[1] else '--'
            if len(seq) >= 4:
                return '-.'
        return '--'
    return '-'


def _extract_axes_data(ax) -> dict:
    """Extract all visible series from one matplotlib Axes."""
    info = {
        'title': ax.get_title() or '',
        'xlabel': ax.get_xlabel() or '',
        'ylabel': ax.get_ylabel() or '',
        'xscale': ax.get_xscale(),
        'yscale': ax.get_yscale(),
        'series': [],
    }

    # Line2D artists
    for line in ax.get_lines():
        label = line.get_label()
        if not label or label.startswith('_'):
            continue
        xd = np.asarray(line.get_xdata(), dtype=float)
        yd = np.asarray(line.get_ydata(), dtype=float)
        if len(xd) == 0:
            continue
        marker_str = str(line.get_marker())
        has_marker = marker_str not in ('None', '', ' ', 'none')
        info['series'].append({
            'type': 'line',
            'xdata': xd, 'ydata': yd,
            'label': label,
            'color': _to_hex(line.get_color()),
            'linewidth': line.get_linewidth(),
            'linestyle': _normalize_ls(line.get_linestyle()),
            'has_marker': has_marker,
            'marker': marker_str,
            'markersize': line.get_markersize(),
        })

    # PathCollection (scatter)
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            continue
        label = coll.get_label()
        if not label or label.startswith('_'):
            continue
        offsets = np.asarray(coll.get_offsets(), dtype=float)
        if offsets.size == 0:
            continue
        fc = coll.get_facecolors()
        color = _to_hex(fc[0]) if len(fc) > 0 else '#000000'
        sizes = coll.get_sizes()
        sz = float(sizes[0]) if len(sizes) > 0 else 20.0
        info['series'].append({
            'type': 'scatter',
            'xdata': offsets[:, 0], 'ydata': offsets[:, 1],
            'label': label,
            'color': color,
            'linewidth': 0,
            'linestyle': '-',
            'has_marker': True,
            'marker': 'o',
            'markersize': np.sqrt(sz),
        })

    return info


def _shared_x(series) -> bool:
    """True if every series shares the same X array."""
    if len(series) < 2:
        return True
    ref = series[0]['xdata']
    return all(len(s['xdata']) == len(ref) and np.allclose(s['xdata'], ref)
               for s in series[1:])


# ── Public entry point ────────────────────────────────────────────────

def export_figure_to_origin(fig, save_path: str) -> None:
    """Export a matplotlib Figure to an Origin .opju project.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    save_path : str
        Full path ending in .opju.

    Raises
    ------
    ImportError  if originpro is not installed.
    RuntimeError if Origin connection fails.
    """
    import originpro as op          # lazy import — fails gracefully

    op.set_show(True)
    op.new()

    # Figure title from suptitle
    fig_title = ''
    if hasattr(fig, '_suptitle') and fig._suptitle is not None:
        fig_title = fig._suptitle.get_text()

    # Collect axes panels that contain plotted data
    panels = []
    for ax in fig.get_axes():
        ax_data = _extract_axes_data(ax)
        if ax_data['series']:
            panels.append(ax_data)

    if not panels:
        raise ValueError("No plottable data found in figure")

    # Create one workbook for the figure
    book = op.new_book('w', lname=fig_title or 'SOLIS Data')

    for pidx, panel in enumerate(panels):
        # ── Worksheet ──
        sheet_name = (panel['title'] or f'Panel {pidx + 1}')[:31]
        sheet_name = sheet_name.replace('/', '_').replace('\\', '_')

        if pidx == 0:
            wks = book[0]
            wks.name = sheet_name
        else:
            wks = book.add_sheet(sheet_name)

        series = panel['series']
        share_x = _shared_x(series)

        # Column layout: shared X or alternating X/Y
        col_map = []  # list of (xcol, ycol) per series
        if share_x:
            x_label = panel['xlabel'] or 'X'
            wks.from_list(0, series[0]['xdata'].tolist(),
                          lname=x_label, axis='X')
            for j, s in enumerate(series):
                wks.from_list(j + 1, s['ydata'].tolist(),
                              lname=s['label'], axis='Y')
                col_map.append((0, j + 1))
        else:
            for j, s in enumerate(series):
                xc, yc = 2 * j, 2 * j + 1
                wks.from_list(xc, s['xdata'].tolist(),
                              lname=panel['xlabel'] or 'X', axis='X')
                wks.from_list(yc, s['ydata'].tolist(),
                              lname=s['label'], axis='Y')
                col_map.append((xc, yc))

        # ── Graph ──
        gp = op.new_graph(lname=panel['title'] or f'Graph {pidx + 1}')
        gl = gp[0]

        for j, s in enumerate(series):
            xcol, ycol = col_map[j]

            # Determine Origin plot type
            if s['type'] == 'scatter':
                ptype = 's'
            elif s['has_marker'] and s['linewidth'] > 0.3:
                ptype = 'linesymb'
            elif s['has_marker']:
                ptype = 's'
            else:
                ptype = 'line'

            p = gl.add_plot(wks, coly=ycol, colx=xcol, type=ptype)

            # Color
            p.color = s['color']

            # Line style + width
            if ptype in ('line', 'linesymb'):
                origin_dash = _LS_MAP.get(s['linestyle'], 0)
                p.set_cmd(f'-d {origin_dash}')
                p.set_cmd(f'-wp {s["linewidth"]:.1f}')

            # Symbol
            if ptype in ('s', 'linesymb'):
                sym = _MARKER_MAP.get(s['marker'], 1)
                p.symbol_kind = sym
                p.symbol_size = max(3, int(s['markersize']))

        # Group for legend
        if len(series) > 1:
            gl.group()

        # Axis titles
        if panel['xlabel']:
            gl.axis('x').title = panel['xlabel']
        if panel['ylabel']:
            gl.axis('y').title = panel['ylabel']

        # Axis scale
        if panel['xscale'] == 'log':
            gl.xscale = 'log10'
        if panel['yscale'] == 'log':
            gl.yscale = 'log10'

        gl.rescale()

    # Save project (leave Origin open)
    op.save(save_path)
    logger.info(f"Origin project saved: {save_path}")
