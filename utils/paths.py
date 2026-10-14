"""Bundled resources are read-only; generated model caches live outside the app."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def cache_root():
    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        return Path.home() / "Library/Caches/Gazeshift"
    return ROOT / ".cache"
