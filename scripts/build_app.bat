@echo off
title Silver Guardian System - Build Script

echo ========================================
echo   Silver Guardian - Application Packager
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
cd /d "%PROJECT_DIR%"

echo [1/5] Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo [2/5] Installing build dependencies...
pip install pyinstaller pywin32
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo [3/5] Creating directory structure...
if not exist "dist" mkdir dist
if not exist "build" mkdir build
if not exist "data" mkdir data
if not exist "data\logs" mkdir data\logs
if not exist "data\alarms" mkdir data\alarms
if not exist "models" mkdir models

echo.
echo [4/5] Fixing pathlib compatibility issue...
pip uninstall pathlib -y >nul 2>&1

echo.
echo [5/5] Starting PyInstaller...
pyinstaller --clean silver_guardian.spec

if %errorlevel% neq 0 (
    echo.
    echo Build FAILED! Please check errors above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo Output: dist\silver_guardian.exe
echo ========================================
echo.

pause
