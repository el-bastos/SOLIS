#!/bin/bash
# Build script for macOS using PyInstaller

echo "========================================"
echo "Building SOLIS for macOS"
echo "========================================"

# Activate virtual environment
source venv/bin/activate

# Install PyInstaller if not present
pip install pyinstaller

# Clean previous builds
rm -rf build dist

# Build with PyInstaller
echo "Building application bundle..."
pyinstaller solis.spec

# Check if build was successful
if [ -d "dist/SOLIS.app" ]; then
    echo ""
    echo "========================================"
    echo "Build successful!"
    echo "Application location: dist/SOLIS.app"
    echo "========================================"

    # Optional: Create DMG
    echo ""
    read -p "Create DMG installer? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating DMG..."
        hdiutil create -volname "SOLIS" -srcfolder dist/SOLIS.app -ov -format UDZO dist/SOLIS-1.0.0-macOS.dmg
        echo "DMG created: dist/SOLIS-1.0.0-macOS.dmg"
    fi
else
    echo ""
    echo "========================================"
    echo "Build failed! Check the output above for errors."
    echo "========================================"
    exit 1
fi
