@echo off
title Silver Guardian System - Full Build

echo ========================================
echo   Silver Guardian - Full Build Process
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
cd /d "%PROJECT_DIR%"

echo [Setup] Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.9 or higher.
    pause
    exit /b 1
)
python --version

echo.
echo [1/5] Updating pip...
python -m pip install --upgrade pip -q

echo.
echo [2/5] Installing project dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo WARNING: Some dependencies failed to install, but continuing...
)

echo.
echo [3/5] Installing build tools...
pip install pyinstaller pywin32 -q

echo.
echo [4/5] Starting build process...
call scripts\build_app.bat

if %errorlevel% neq 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo [5/5] Preparing release package...
call scripts\prepare_release.bat

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Full build process completed!
    echo Release package: SilverGuardian_Release
    echo ========================================
    echo.
    echo You can zip the "SilverGuardian_Release" directory for distribution.
)

echo.
pause
