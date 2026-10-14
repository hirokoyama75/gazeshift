#!/bin/bash
set -eu
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "Gazeshift requires macOS 14+ on Apple Silicon. Run Terminal without Rosetta."
    read -r -p "Press Return to close. " _
    exit 1
fi
if [ "$(sw_vers -productVersion | cut -d. -f1)" -lt 14 ]; then
    echo "Gazeshift requires macOS 14 or later."
    read -r -p "Press Return to close. " _
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    read -r -p "Press Return to close. " _
    exit 1
fi
export UV_PROJECT_ENVIRONMENT="$PWD/.cache/mac"
echo "Preparing Gazeshift. Allow Terminal/Python camera access when prompted."
if ! uv sync --locked --extra virtualcam; then
    echo "Setup failed. Check your internet connection and try again."
    read -r -p "Press Return to close. " _
    exit 1
fi
uv run --locked --exact --extra virtualcam gui.py
