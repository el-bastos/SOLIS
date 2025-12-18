@echo off
REM Build script for Windows using PyInstaller

echo ========================================
echo Building SOLIS for Windows
echo ========================================

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install PyInstaller if not present
pip install pyinstaller

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build with PyInstaller
echo Building executable...
pyinstaller solis.spec

REM Check if build was successful
if exist dist\SOLIS\SOLIS.exe (
    echo.
    echo ========================================
    echo Build successful!
    echo Executable location: dist\SOLIS\SOLIS.exe
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Build failed! Check the output above for errors.
    echo ========================================
    exit /b 1
)

pause
