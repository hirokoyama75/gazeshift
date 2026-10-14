@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>&1
if errorlevel 1 (
    echo Install uv first: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)
set "UV_PROJECT_ENVIRONMENT=%~dp0.cache\cuda"
echo Preparing NVIDIA support, including CUDA and cuDNN. First download is large...
uv sync --locked --no-group cpu --extra cuda --extra virtualcam
if errorlevel 1 (
    echo NVIDIA setup failed. The regular Start Gazeshift launcher still uses CPU.
    pause
    exit /b 1
)
start "" ".cache\cuda\Scripts\pythonw.exe" gui.py --provider cuda
