"""Offline package checks: real models, Tk and backends, with camera I/O blocked."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

from utils.paths import ROOT, cache_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {"status": "FAIL", "frozen": bool(getattr(sys, "frozen", False)),
              "architecture": platform.machine(), "cache": str(cache_root()),
              "manual_pending": ["physical camera and OBS", "sustained FPS, CPU, GPU, memory and power"]}
    try:
        info = json.loads((ROOT / "build-info.json").read_text())
        for name, expected in info["models"].items():
            if hashlib.sha256((ROOT / "weights" / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Bundled model is missing or corrupt: {name}")
        report["models"] = info["models"]
        import cv2
        import pyvirtualcam
        import onnxruntime as ort
        def forbidden(*args, **kwargs):
            raise AssertionError("Bundle smoke check attempted camera I/O")
        cv2.VideoCapture = pyvirtualcam.Camera = forbidden
        from engine.geometry import CameraGeometry
        from engine.landmark_tracker import LandmarkTracker
        import numpy as np
        tracker = LandmarkTracker(CameraGeometry())
        try:
            assert tracker.track(np.zeros((720,1280,3), np.uint8), 1) is None
        finally:
            tracker.close()
        report["tracker"] = "PASS"
        import tkinter as tk
        import gui
        gui.list_cameras = lambda: []
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                app = gui.Desktop(root, Path(directory)/"desktop.json", Path(directory)/"config.json")
                root.update_idletasks()
                assert app.worker is None
                report["tk"] = str(root.tk.call("info", "patchlevel"))
                if root.tk.call("tk", "windowingsystem") == "aqua":
                    root.tk.call("::tk::mac::Quit")
                    report["mac_quit"] = "PASS"
                else:
                    app.close()
            except BaseException:
                root.destroy()
                raise
        from scripts.benchmark_provider import main as benchmark
        report["backends"] = {}
        for key, name in (("cpu", "CPUExecutionProvider"), ("coreml", "CoreMLExecutionProvider")):
            if name not in ort.get_available_providers():
                continue
            path = args.report.with_name(f"{args.report.stem}-{key}.json")
            if benchmark(["--provider", key, "--report", str(path)]):
                raise RuntimeError(f"Bundled {key} benchmark/accuracy check failed: {path}")
            report["backends"][key] = json.loads(path.read_text())
        if "cpu" not in report["backends"]:
            raise RuntimeError("Bundled CPU provider missing")
        report["status"] = "PASS"
    except Exception:
        import traceback
        report["error"] = traceback.format_exc()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1
