@echo off
chcp 65001 >nul
title Silver Guardian System
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ========================================
echo    Silver Guardian System
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Checking system environment...
if not exist "config\config.yaml" (
    echo ERROR: Configuration file not found!
    echo Please ensure config\config.yaml exists.
    echo.
    pause
    exit /b 1
)

if not exist "silver_guardian.exe" (
    echo ERROR: silver_guardian.exe not found!
    echo Please build the application first by running:
    echo   scripts\full_build.bat
    echo or:
    echo   scripts\build_app.bat
    echo.
    pause
    exit /b 1
)

if not exist "data" mkdir data
if not exist "data\logs" mkdir data\logs
if not exist "data\alarms" mkdir data\alarms
if not exist "models" mkdir models

echo.
echo [2/3] Preparing to start system...
echo.
echo Startup options:
echo 1. Demo mode (recommended, no camera required)
echo 2. Production mode (connect real cameras)
echo 3. Exit
echo.

set /p choice=Please select startup mode (1-3):

if "%choice%"=="1" (
    echo.
    echo [3/3] Starting in demo mode...
    echo.
    "silver_guardian.exe" --demo
) else if "%choice%"=="2" (
    echo.
    echo [3/3] Starting in production mode...
    echo.
    "silver_guardian.exe"
) else if "%choice%"=="3" (
    echo.
    echo Exiting.
    exit /b 0
) else (
    echo.
    echo Invalid selection, starting in demo mode by default...
    "silver_guardian.exe" --demo
)

if %errorlevel% neq 0 (
    echo.
    echo Program exited with error code: %errorlevel%
    echo Please check log file data\logs\system.log
    echo.
    pause
)
