@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>&1
if errorlevel 1 (
    echo Install uv first: https://docs.astral.sh/uv/getting-started/installation/
    echo Then double-click this file again.
    pause
    exit /b 1
)
echo Preparing gazeshift. First launch may take a little longer...
uv sync --extra virtualcam
if errorlevel 1 (
    echo Setup failed. Check your internet connection and try again.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" gui.py
