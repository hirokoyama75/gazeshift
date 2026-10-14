"""Windowed app entry point, plus an explicit camera-free build smoke check."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    if "--self-test" in sys.argv:
        from scripts.check_bundle import main as check
        return check()
    try:
        from gui import main as desktop
        desktop()
        return 0
    except Exception:
        import traceback
        from tkinter import messagebox
        log = Path.home() / ".gaze_cam/startup-error.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc())
        messagebox.showerror("Gazeshift could not start", f"Startup failed. Details were saved to:\n{log}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
