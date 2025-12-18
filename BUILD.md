# Building SOLIS from Source

This guide covers building SOLIS into standalone executables for Windows, macOS, and Linux.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Build](#quick-build)
- [Platform-Specific Instructions](#platform-specific-instructions)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Manual Build Process](#manual-build-process)
- [Troubleshooting](#troubleshooting)
- [GitHub Actions (Automated Builds)](#github-actions-automated-builds)

---

## Prerequisites

### All Platforms
- Python 3.13 (Numba requires <3.14)
- Git
- At least 2GB of free disk space

### Platform-Specific Requirements

**Windows:**
- Visual C++ Redistributable (usually included with Python)

**macOS:**
- Xcode Command Line Tools: `xcode-select --install`
- For DMG creation: pre-installed `hdiutil`

**Linux:**
- Development libraries:
  ```bash
  sudo apt-get install -y libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0
  ```
- For AppImage creation: `appimagetool` (optional)

---

## Quick Build

### 1. Clone and Setup

```bash
git clone https://github.com/el-bastos/SOLIS.git
cd SOLIS
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
.\setup_venv.ps1
```

**Windows (CMD):**
```cmd
setup_venv.bat
```

**Linux/macOS:**
```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Build

**Windows:**
```cmd
build_scripts\build_windows.bat
```

**macOS:**
```bash
bash build_scripts/build_macos.sh
```

**Linux:**
```bash
bash build_scripts/build_linux.sh
```

---

## Platform-Specific Instructions

### Windows

#### Building EXE

1. **Activate virtual environment:**
   ```cmd
   venv\Scripts\activate
   ```

2. **Install PyInstaller:**
   ```cmd
   pip install pyinstaller
   ```

3. **Build:**
   ```cmd
   pyinstaller solis.spec
   ```

4. **Result:**
   - Executable: `dist\SOLIS\SOLIS.exe`
   - All dependencies included in `dist\SOLIS\` folder

#### Creating Installer (Optional)

Use [Inno Setup](https://jrsoftware.org/isinfo.php) or [NSIS](https://nsis.sourceforge.io/) to create an installer:

**Example Inno Setup script:**
```inno
[Setup]
AppName=SOLIS
AppVersion=1.0.0
DefaultDirName={pf}\SOLIS
DefaultGroupName=SOLIS
OutputBaseFilename=SOLIS-1.0.0-Windows-Setup

[Files]
Source: "dist\SOLIS\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\SOLIS"; Filename: "{app}\SOLIS.exe"
```

---

### macOS

#### Building APP Bundle

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

3. **Build:**
   ```bash
   pyinstaller solis.spec
   ```

4. **Result:**
   - Application bundle: `dist/SOLIS.app`

#### Creating DMG (Optional)

```bash
hdiutil create -volname "SOLIS" -srcfolder dist/SOLIS.app -ov -format UDZO dist/SOLIS-1.0.0-macOS.dmg
```

#### Code Signing (Optional)

For distribution outside App Store:
```bash
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" dist/SOLIS.app
```

For notarization:
```bash
xcrun notarytool submit dist/SOLIS-1.0.0-macOS.dmg --apple-id "your@email.com" --password "app-specific-password" --team-id "TEAMID"
```

---

### Linux

#### Building Executable

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

3. **Build:**
   ```bash
   pyinstaller solis.spec
   ```

4. **Result:**
   - Executable: `dist/SOLIS/SOLIS`
   - All dependencies included in `dist/SOLIS/` folder

#### Creating AppImage (Optional)

1. **Download appimagetool:**
   ```bash
   wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
   chmod +x appimagetool-x86_64.AppImage
   ```

2. **Create AppDir structure:**
   ```bash
   mkdir -p SOLIS.AppDir/usr/bin
   mkdir -p SOLIS.AppDir/usr/share/applications
   mkdir -p SOLIS.AppDir/usr/share/icons/hicolor/256x256/apps

   cp -r dist/SOLIS/* SOLIS.AppDir/usr/bin/
   ```

3. **Create desktop file** (`SOLIS.AppDir/usr/share/applications/solis.desktop`):
   ```ini
   [Desktop Entry]
   Type=Application
   Name=SOLIS
   Comment=Singlet Oxygen Luminescence Investigation System
   Exec=SOLIS
   Icon=solis
   Categories=Science;Education;
   Terminal=false
   ```

4. **Copy icon:**
   ```bash
   convert utils/SOLIS_logo.jpg SOLIS.AppDir/usr/share/icons/hicolor/256x256/apps/solis.png
   ```

5. **Create AppRun** (`SOLIS.AppDir/AppRun`):
   ```bash
   #!/bin/bash
   SELF=$(readlink -f "$0")
   HERE=${SELF%/*}
   export PATH="${HERE}/usr/bin/:${PATH}"
   export LD_LIBRARY_PATH="${HERE}/usr/lib/:${LD_LIBRARY_PATH}"
   exec "${HERE}/usr/bin/SOLIS" "$@"
   ```

   ```bash
   chmod +x SOLIS.AppDir/AppRun
   ```

6. **Build AppImage:**
   ```bash
   ./appimagetool-x86_64.AppImage SOLIS.AppDir SOLIS-1.0.0-x86_64.AppImage
   ```

#### Creating DEB Package (Optional)

```bash
# Create package structure
mkdir -p SOLIS-deb/DEBIAN
mkdir -p SOLIS-deb/usr/local/bin
mkdir -p SOLIS-deb/usr/share/applications
mkdir -p SOLIS-deb/usr/share/icons

# Copy files
cp -r dist/SOLIS SOLIS-deb/usr/local/bin/

# Create control file
cat > SOLIS-deb/DEBIAN/control << EOF
Package: solis
Version: 1.0.0
Architecture: amd64
Maintainer: Erick Leite Bastos <elbastos@iq.usp.br>
Description: Singlet Oxygen Luminescence Investigation System
 A comprehensive GUI application for analyzing singlet oxygen decay kinetics.
EOF

# Build package
dpkg-deb --build SOLIS-deb
mv SOLIS-deb.deb SOLIS-1.0.0-amd64.deb
```

---

## Manual Build Process

If automated scripts don't work, follow these steps:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install pyinstaller
```

### 2. Clean Previous Builds

```bash
rm -rf build dist *.spec
```

### 3. Run PyInstaller

**Basic command:**
```bash
pyinstaller --name=SOLIS \
    --windowed \
    --add-data="utils/SOLIS_logo.jpg:utils" \
    --add-data="utils/SOLIS_logo.svg:utils" \
    --add-data="data:data" \
    --hidden-import=PyQt6.QtCore \
    --hidden-import=PyQt6.QtGui \
    --hidden-import=PyQt6.QtWidgets \
    --hidden-import=numpy \
    --hidden-import=scipy \
    --hidden-import=pandas \
    --hidden-import=matplotlib \
    --hidden-import=numba \
    solis_gui.py
```

**Or use the spec file:**
```bash
pyinstaller solis.spec
```

---

## Troubleshooting

### Import Errors

**Problem:** Missing modules when running executable

**Solution:** Add hidden imports to `solis.spec`:
```python
hiddenimports = [
    'missing_module_name',
]
```

### File Not Found Errors

**Problem:** Resources (images, data files) not found

**Solution:** Add to `datas` in `solis.spec`:
```python
datas = [
    ('path/to/file', 'destination/folder'),
]
```

### Numba Compilation Errors

**Problem:** Numba functions fail in executable

**Solution:** Numba should work with ahead-of-time compilation. If issues persist:
1. Ensure Python version is 3.10-3.13 (Numba does NOT support 3.14+)
2. Update Numba: `pip install --upgrade numba`
3. On macOS Apple Silicon, use conda: `conda install -c conda-forge numba`

### PDF Export Not Working

**Problem:** "No module named matplotlib.backends.backend_pdf"

**Solution:** This was fixed in Session 55. Ensure `solis.spec` includes:
```python
'matplotlib.backends.backend_pdf',
'matplotlib.pyplot',
'scipy.signal',
```

### Large Executable Size

**Problem:** Executable is very large (>500MB)

**Solutions:**
1. Use UPX compression (enabled by default in spec file)
2. Exclude unnecessary modules in `solis.spec`:
   ```python
   excludes=['tkinter', 'test', 'unittest']
   ```
3. Use `--onefile` mode (slower startup but single file)

### Windows Antivirus False Positives

**Problem:** Antivirus flags executable as malware

**Solutions:**
1. Code sign the executable
2. Submit false positive report to antivirus vendor
3. Distribute as source code with instructions

### macOS "App is damaged" Error

**Problem:** macOS blocks unsigned app

**Solution:** Users can run:
```bash
xattr -cr /path/to/SOLIS.app
```

Or you can code sign and notarize the app.

---

## GitHub Actions (Automated Builds)

SOLIS includes GitHub Actions workflows for automated builds:

### Triggering Builds

**Automatic:**
- On push to `master`, `main`, or `develop` branches
- On pull requests
- On version tags (e.g., `v1.0.0`)

**Manual:**
1. Go to repository on GitHub
2. Click "Actions" tab
3. Select "Build SOLIS" workflow
4. Click "Run workflow"

### Downloading Artifacts

1. Go to "Actions" tab
2. Click on a successful workflow run
3. Download platform-specific artifacts:
   - `SOLIS-Windows`
   - `SOLIS-macOS`
   - `SOLIS-Linux`

### Creating Releases

1. Tag a commit:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

2. GitHub Actions will automatically:
   - Build for all platforms
   - Create a GitHub Release
   - Attach executables to the release

---

## Build Size Reference

Expected sizes (approximate):

| Platform | Size |
|----------|------|
| Windows EXE (folder) | 400-600 MB |
| macOS APP bundle | 400-600 MB |
| macOS DMG | 200-300 MB |
| Linux folder | 400-600 MB |
| Linux AppImage | 200-300 MB |

---

## Additional Resources

- [PyInstaller Documentation](https://pyinstaller.org/)
- [PyQt6 Deployment Guide](https://www.riverbankcomputing.com/static/Docs/PyQt6/deployment.html)
- [AppImage Documentation](https://docs.appimage.org/)
- [Inno Setup Documentation](https://jrsoftware.org/ishelp/)

---

## Questions or Issues?

If you encounter problems during the build process:

1. Check [GitHub Issues](https://github.com/el-bastos/SOLIS/issues)
2. Create a new issue with:
   - Your operating system and version
   - Python version (`python --version`)
   - Complete error message
   - Build command used

---

**Last updated:** 2025-12-02
**SOLIS Version:** 1.0.0-beta
