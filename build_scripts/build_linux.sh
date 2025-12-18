#!/bin/bash
# Build script for Linux using PyInstaller

echo "========================================"
echo "Building SOLIS for Linux"
echo "========================================"

# Activate virtual environment
source venv/bin/activate

# Install PyInstaller if not present
pip install pyinstaller

# Clean previous builds
rm -rf build dist

# Build with PyInstaller
echo "Building executable..."
pyinstaller solis.spec

# Check if build was successful
if [ -f "dist/SOLIS/SOLIS" ]; then
    echo ""
    echo "========================================"
    echo "Build successful!"
    echo "Executable location: dist/SOLIS/SOLIS"
    echo "========================================"

    # Optional: Create AppImage
    echo ""
    read -p "Create AppImage? (requires appimagetool) (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating AppImage structure..."

        # Create AppDir structure
        mkdir -p SOLIS.AppDir/usr/bin
        mkdir -p SOLIS.AppDir/usr/share/applications
        mkdir -p SOLIS.AppDir/usr/share/icons/hicolor/256x256/apps

        # Copy files
        cp -r dist/SOLIS/* SOLIS.AppDir/usr/bin/

        # Create desktop file
        cat > SOLIS.AppDir/usr/share/applications/solis.desktop << EOF
[Desktop Entry]
Type=Application
Name=SOLIS
Comment=Singlet Oxygen Luminescence Investigation System
Exec=SOLIS
Icon=solis
Categories=Science;Education;
Terminal=false
EOF

        # Copy icon (convert jpg to png if needed)
        if command -v convert &> /dev/null; then
            convert utils/SOLIS_logo.jpg SOLIS.AppDir/usr/share/icons/hicolor/256x256/apps/solis.png
        else
            cp utils/SOLIS_logo.jpg SOLIS.AppDir/solis.png
        fi

        # Create AppRun
        cat > SOLIS.AppDir/AppRun << 'EOF'
#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${SELF%/*}
export PATH="${HERE}/usr/bin/:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib/:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/SOLIS" "$@"
EOF
        chmod +x SOLIS.AppDir/AppRun

        # Build AppImage (requires appimagetool)
        if command -v appimagetool &> /dev/null; then
            appimagetool SOLIS.AppDir SOLIS-1.0.0-x86_64.AppImage
            echo "AppImage created: SOLIS-1.0.0-x86_64.AppImage"
        else
            echo "appimagetool not found. Please install it to create AppImage."
            echo "AppDir structure created in SOLIS.AppDir/"
        fi
    fi
else
    echo ""
    echo "========================================"
    echo "Build failed! Check the output above for errors."
    echo "========================================"
    exit 1
fi
