@echo off
title Silver Guardian - Release Preparer

echo ========================================
echo   Silver Guardian - Release Preparer
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
cd /d "%PROJECT_DIR%"

set RELEASE_DIR=SilverGuardian_Release

echo [1/6] Creating release directory...
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
mkdir "%RELEASE_DIR%"

echo.
echo [2/6] Copying main program and config...
if not exist "dist\silver_guardian.exe" (
    echo ERROR: Build not found at dist\silver_guardian.exe
    echo Please run scripts\build_app.bat first
    pause
    exit /b 1
)

copy "dist\silver_guardian.exe" "%RELEASE_DIR%\"
xcopy /e /i /y "config" "%RELEASE_DIR%\config"

echo.
echo [3/6] Creating data directories...
mkdir "%RELEASE_DIR%\data"
mkdir "%RELEASE_DIR%\data\logs"
mkdir "%RELEASE_DIR%\data\alarms"
mkdir "%RELEASE_DIR%\data\alarms\screenshots"
mkdir "%RELEASE_DIR%\data\alarms\videos"
mkdir "%RELEASE_DIR%\models"

echo.
echo [4/6] Copying startup scripts and docs...
copy "启动应用.bat" "%RELEASE_DIR%\"
copy "配置说明.txt" "%RELEASE_DIR%\"
copy "使用说明.txt" "%RELEASE_DIR%\"
copy "README_PACKAGE.md" "%RELEASE_DIR%\"

echo.
echo [5/6] Creating placeholder files...
echo. > "%RELEASE_DIR%\data\logs\system.log"
echo Place model files in this directory > "%RELEASE_DIR%\models\README.txt"

echo.
echo [6/6] Cleaning up temporary files...
if exist "%RELEASE_DIR%\*.pyc" del /f /q "%RELEASE_DIR%\*.pyc"
if exist "%RELEASE_DIR%\__pycache__" rmdir /s /q "%RELEASE_DIR%\__pycache__"

echo.
echo ========================================
echo Release package ready!
echo Directory: %RELEASE_DIR%
echo ========================================
echo.
echo You can zip this directory for distribution.
echo Users can run 启动应用.bat after extracting.
echo.

dir "%RELEASE_DIR%"

echo.
pause
