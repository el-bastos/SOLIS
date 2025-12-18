# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SOLIS
Supports Windows, macOS, and Linux builds
Updated: 2025-12-02 (Session 55 - Added missing hiddenimports)

IMPORTANT: Build from venv with Python 3.13.x for Numba support!
    cd "path/to/SOLIS_CLEAN"
    venv/Scripts/python -m PyInstaller SOLIS.spec --clean
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all data files
datas = [
    ('utils/SOLIS_logo.jpg', 'utils'),
    ('utils/SOLIS_logo.svg', 'utils'),
    ('logo/SOLIS-ICON.png', 'logo'),
    ('logo/SOLIS-LOGO.png', 'logo'),
    ('icons', 'icons'),  # SVG icons for GUI (Session 47)
    ('help.json', '.'),  # Help system content (Session 45)
    ('examples', 'examples'),  # Include example data
    ('docs', 'docs'),  # Include documentation
]

# Collect hidden imports
hiddenimports = [
    # PyQt6
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    # Scientific computing
    'numpy',
    'scipy',
    'scipy.optimize',
    'scipy.special',
    'scipy.integrate',
    'scipy.stats',
    'scipy.signal',  # Used in spike detection
    'pandas',
    # Plotting (matplotlib only - Plotly migration complete)
    'matplotlib',
    'matplotlib.pyplot',  # Used by all plotting modules
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_pdf',  # Required for PDF export
    'matplotlib.figure',
    'matplotlib.gridspec',
    # Performance (Numba auto-detected if present, has fallback)
    # 'numba',  # Optional - removed, not needed for PyInstaller
    # 'llvmlite',  # Optional - removed, not needed for PyInstaller
    # Data structures
    'dataclasses',
    # 'dataclasses_json',  # Not used in SOLIS - removed
    'typing',
    'typing_extensions',
    # Standard library
    'pathlib',
    'unittest',
    'unittest.mock',
    # 'collections.abc',  # Standard library, auto-included - removed
]

# Add all submodules from our packages
hiddenimports += collect_submodules('core')
hiddenimports += collect_submodules('data')
hiddenimports += collect_submodules('heterogeneous')
hiddenimports += collect_submodules('gui')
hiddenimports += collect_submodules('plotting')
hiddenimports += collect_submodules('utils')
hiddenimports += collect_submodules('surplus')

a = Analysis(
    ['show_splash_then_load.py'],  # CORRECT ENTRY POINT with splash screen
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'test',
        # Don't exclude unittest - needed by setuptools._distutils.compilers.C.msvc (unittest.mock)
        # Don't exclude email/http - some packages might need them
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Platform-specific configurations
if sys.platform == 'win32':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='SOLIS',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,  # Hide console window for production
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='logo/SOLIS-ICON.png',  # Application icon
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='SOLIS',
    )

elif sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='SOLIS',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='SOLIS',
    )

    app = BUNDLE(
        coll,
        name='SOLIS.app',
        icon='logo/SOLIS-ICON.png',  # Application icon
        bundle_identifier='br.usp.iq.solis',
        info_plist={
            'CFBundleName': 'SOLIS',
            'CFBundleDisplayName': 'SOLIS',
            'CFBundleGetInfoString': 'Singlet Oxygen Luminescence Investigation System',
            'CFBundleIdentifier': 'br.usp.iq.solis',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'LSMinimumSystemVersion': '10.13.0',
            'NSRequiresAquaSystemAppearance': False,
            'LSApplicationCategoryType': 'public.app-category.education',
        },
    )

else:  # Linux
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='SOLIS',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='SOLIS',
    )
