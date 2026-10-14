@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>&1
if errorlevel 1 (
    echo Install uv first: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)
set "UV_PROJECT_ENVIRONMENT=%~dp0.cache\dml"
echo Preparing AMD / Intel GPU support. First launch may take a little longer...
uv sync --no-group cpu --extra dml --extra virtualcam
if errorlevel 1 (
    echo GPU setup failed. The regular Start Gazeshift launcher still uses CPU.
    pause
    exit /b 1
)
start "" ".cache\dml\Scripts\pythonw.exe" gui.py --provider dml
