#!/usr/bin/env python3
"""
Centralized Stylesheet Definitions for SOLIS GUI

This module contains all QStyleSheet definitions used throughout the SOLIS
application. Centralizing styles ensures consistency and makes theming easier.

Color Scheme:
- SOLIS Orange: #FF8C00
- Selection Color: #FFE4B5 (light orange/peach)
- Button Hover: #ADD8E6 (light blue)
- Gray Text: #666666
- Section Background: #F0F0F0 (light gray)
- Border Color: #999999

Design Principles (from Session 48):
- NO hover effects on trees and tables (hover: transparent)
- Consistent selection highlighting (#FFE4B5)
- Clean, professional appearance
"""

# =============================================================================
# SOLIS Theme Colors
# =============================================================================

SOLIS_ORANGE = "#FF8C00"
SELECTION_BG = "#FFE4B5"  # Light orange/peach for selected items
SELECTION_FG = "black"    # Black text on selected items
BUTTON_HOVER = "#ADD8E6"  # Light blue for button hover states
INFO_TEXT = "#666666"     # Gray for info labels
SECTION_BG = "#F0F0F0"    # Light gray for section headers
BORDER_COLOR = "#999999"  # Border color for frames and widgets
WHITE_BG = "#FFFFFF"      # White background


# =============================================================================
# Tree Widget Styles
# =============================================================================

TREE_STYLE = """
    QTreeWidget {
        outline: none;
    }
    QTreeWidget::item {
        background-color: transparent;
    }
    QTreeWidget::item:hover {
        background-color: transparent;
        border: none;
    }
    QTreeWidget::item:selected {
        background-color: #FFE4B5;
        color: black;
    }
    QTreeWidget::item:selected:hover {
        background-color: #FFE4B5;
        color: black;
    }
"""

# NOTE: No branch styling = platform-native expand/collapse arrows remain visible
# ::item has no hover, only ::branch shows platform default behavior

# Alternative: Tree with no special styling (for dialogs, minimal UI)
TREE_STYLE_MINIMAL = """
    QTreeWidget::item:selected {
        background-color: #FFE4B5;
        color: black;
    }
"""


# =============================================================================
# Table Widget Styles
# =============================================================================

TABLE_STYLE = """
    QTableWidget {
        outline: none;
    }
    QTableWidget::item {
        background-color: transparent;
    }
    QTableWidget::item:hover {
        background-color: transparent;
        border: none;
    }
    QTableWidget::item:selected {
        background-color: #FFE4B5;
        color: black;
    }
    QTableWidget::item:selected:hover {
        background-color: #FFE4B5;
        color: black;
    }
"""

# Alternative: Table with no special styling
TABLE_STYLE_MINIMAL = """
    QTableWidget::item:selected {
        background-color: #FFE4B5;
        color: black;
    }
"""


# =============================================================================
# Tab Widget Styles
# =============================================================================

def get_tab_close_button_style(icon_path: str) -> str:
    """
    Get tab widget stylesheet with custom close button icon.

    Args:
        icon_path: Path to close icon SVG file

    Returns:
        Stylesheet string with embedded icon path
    """
    # Convert Windows backslashes to forward slashes for CSS
    icon_path_css = str(icon_path).replace('\\', '/')

    return f"""
        QTabBar::close-button {{
            image: url({icon_path_css});
            subcontrol-position: right;
        }}
        QTabBar::close-button:hover {{
            background-color: {BUTTON_HOVER};
        }}
    """


# =============================================================================
# Menu Bar Styles
# =============================================================================

MENU_STYLE = """
    QMenu {
        border: 1px solid #999;
        border-radius: 0px;
    }
    QMenu::item {
        padding: 4px 20px 4px 20px;
    }
    QMenu::item:selected {
        background-color: #FFE4B5;
    }
"""


# =============================================================================
# Progress Bar Styles
# =============================================================================

PROGRESS_BAR_STYLE = """
    QProgressBar {
        border: 1px solid #999;
        border-radius: 3px;
        text-align: center;
        background-color: #f0f0f0;
    }
    QProgressBar::chunk {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #90EE90,
            stop: 1 #32CD32
        );
        border-radius: 2px;
    }
"""


# =============================================================================
# Button Styles (Flat, no shadows)
# =============================================================================

BUTTON_STYLE_FLAT = """
    QPushButton {
        background-color: #f0f0f0;
        border: 1px solid #999;
        border-radius: 3px;
        padding: 4px 12px;
        min-height: 20px;
    }
    QPushButton:hover {
        background-color: #e0e0e0;
        border: 1px solid #666;
    }
    QPushButton:pressed {
        background-color: #d0d0d0;
        border: 1px solid #555;
    }
    QPushButton:checked {
        background-color: #FFE4B5;
        border: 1px solid #FF8C00;
    }
    QPushButton:disabled {
        background-color: #f5f5f5;
        border: 1px solid #ccc;
        color: #999;
    }
"""

# Smaller flat button for toolbars and controls
BUTTON_STYLE_FLAT_SMALL = """
    QPushButton {
        background-color: #f0f0f0;
        border: 1px solid #999;
        border-radius: 2px;
        padding: 2px 8px;
        min-height: 18px;
        font-size: 9pt;
    }
    QPushButton:hover {
        background-color: #e0e0e0;
        border: 1px solid #666;
    }
    QPushButton:pressed {
        background-color: #d0d0d0;
    }
    QPushButton:checked {
        background-color: #FFE4B5;
        border: 1px solid #FF8C00;
    }
"""


# =============================================================================
# GroupBox Styles (Flat, no shadows)
# =============================================================================

GROUPBOX_STYLE_FLAT = """
    QGroupBox {
        border: 1px solid #ccc;
        border-radius: 3px;
        margin-top: 8px;
        padding-top: 8px;
        background-color: transparent;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        color: #333;
    }
"""


# =============================================================================
# SpinBox Styles (Flat, no shadows)
# =============================================================================

SPINBOX_STYLE_FLAT = """
    QSpinBox, QDoubleSpinBox {
        border: 1px solid #999;
        border-radius: 2px;
        padding: 2px 4px;
        background-color: white;
    }
    QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #FF8C00;
    }
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        border: none;
        background-color: #f0f0f0;
    }
    QSpinBox::up-button:hover, QSpinBox::down-button:hover,
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #e0e0e0;
    }
"""


# =============================================================================
# ComboBox Styles (Flat, no shadows)
# =============================================================================

COMBOBOX_STYLE_FLAT = """
    QComboBox {
        border: 1px solid #999;
        border-radius: 2px;
        padding: 2px 8px;
        background-color: white;
    }
    QComboBox:focus {
        border: 1px solid #FF8C00;
    }
    QComboBox::drop-down {
        border: none;
        background-color: #f0f0f0;
        width: 20px;
    }
    QComboBox::drop-down:hover {
        background-color: #e0e0e0;
    }
    QComboBox QAbstractItemView {
        border: 1px solid #999;
        selection-background-color: #FFE4B5;
        selection-color: black;
    }
"""


# =============================================================================
# Label Styles (Info Text)
# =============================================================================

INFO_LABEL_STYLE = "color: #666; font-size: 10pt;"
INFO_LABEL_STYLE_SMALL = "color: gray; font-size: 9pt;"
INFO_LABEL_STYLE_WITH_MARGIN = "color: gray; font-size: 9pt; margin: 10px;"


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_tree_style(tree_widget):
    """Apply standard tree widget style (no hover, SOLIS selection)."""
    tree_widget.setStyleSheet(TREE_STYLE)


def apply_table_style(table_widget):
    """Apply standard table widget style (no hover, SOLIS selection)."""
    table_widget.setStyleSheet(TABLE_STYLE)


def apply_menu_style(menu_bar):
    """Apply standard menu bar style."""
    menu_bar.setStyleSheet(MENU_STYLE)


def apply_progress_bar_style(progress_bar):
    """Apply standard progress bar style (green gradient)."""
    progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)


def apply_info_label_style(label, with_margin=False):
    """
    Apply info label style (gray text).

    Args:
        label: QLabel to style
        with_margin: If True, adds 10px margin
    """
    if with_margin:
        label.setStyleSheet(INFO_LABEL_STYLE_WITH_MARGIN)
    else:
        label.setStyleSheet(INFO_LABEL_STYLE)


def apply_flat_button_style(button, small=False):
    """Apply flat button style (no shadows)."""
    if small:
        button.setStyleSheet(BUTTON_STYLE_FLAT_SMALL)
    else:
        button.setStyleSheet(BUTTON_STYLE_FLAT)


def apply_flat_groupbox_style(groupbox):
    """Apply flat groupbox style (no shadows)."""
    groupbox.setStyleSheet(GROUPBOX_STYLE_FLAT)


def apply_flat_spinbox_style(spinbox):
    """Apply flat spinbox style (no shadows)."""
    spinbox.setStyleSheet(SPINBOX_STYLE_FLAT)


def apply_flat_combobox_style(combobox):
    """Apply flat combobox style (no shadows)."""
    combobox.setStyleSheet(COMBOBOX_STYLE_FLAT)


def get_global_flat_style():
    """
    Get a global stylesheet for the entire application that removes all shadows.
    Apply this to QApplication to affect all widgets.
    """
    return f"""
        /* Flat buttons - no shadows */
        {BUTTON_STYLE_FLAT}

        /* Flat groupboxes */
        {GROUPBOX_STYLE_FLAT}

        /* Flat spinboxes */
        {SPINBOX_STYLE_FLAT}

        /* Flat comboboxes */
        {COMBOBOX_STYLE_FLAT}

        /* Flat line edits */
        QLineEdit {{
            border: 1px solid #999;
            border-radius: 2px;
            padding: 2px 4px;
            background-color: white;
        }}
        QLineEdit:focus {{
            border: 1px solid #FF8C00;
        }}

        /* Flat tool buttons */
        QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 2px;
            padding: 2px;
        }}
        QToolButton:hover {{
            background-color: #e0e0e0;
            border: 1px solid #999;
        }}
        QToolButton:pressed {{
            background-color: #d0d0d0;
        }}
        QToolButton:checked {{
            background-color: #FFE4B5;
            border: 1px solid #FF8C00;
        }}

        /* Flat scroll bars */
        QScrollBar:vertical {{
            border: none;
            background-color: #f0f0f0;
            width: 12px;
        }}
        QScrollBar::handle:vertical {{
            background-color: #c0c0c0;
            border-radius: 4px;
            min-height: 20px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: #a0a0a0;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            border: none;
            background-color: #f0f0f0;
            height: 12px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: #c0c0c0;
            border-radius: 4px;
            min-width: 20px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: #a0a0a0;
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
    """


# =============================================================================
# Migration Notes
# =============================================================================

"""
MIGRATION GUIDE FOR EXISTING CODE:

Old code:
    tree.setStyleSheet('''
        QTreeWidget::item:hover {
            background-color: transparent;
        }
        QTreeWidget::item:selected {
            background-color: #FFE4B5;
            color: black;
        }
    ''')

New code:
    from gui.stylesheets import apply_tree_style
    apply_tree_style(tree)

Or:
    from gui.stylesheets import TREE_STYLE
    tree.setStyleSheet(TREE_STYLE)

Benefits:
- Consistent styling across all widgets
- Single source of truth for colors
- Easy to change theme (modify this file only)
- Less code duplication
- Better maintainability
"""
