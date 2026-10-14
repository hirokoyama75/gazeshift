"""Build and smoke-check a self-contained Apple Silicon app on macOS."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.assets import fetch

MODELS = {
    "flx_12.onnx": "cc335a395ab17261fb36847d29291523fb5bbee10ec5b2395e9b83367ca1bf66",
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
}
FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def check_idle(executable, output):
    """Measure the normal packaged interface with its camera left off."""
    def sample(pid):
        value = subprocess.check_output(["ps", "-p", str(pid), "-o", "time=,rss="], text=True).split()
        seconds = sum(float(part)*60**i for i, part in enumerate(reversed(value[0].split(":"))))
        return seconds, int(value[1])/1024
    with tempfile.TemporaryDirectory(dir=output, prefix=".idle-") as directory, (output / "idle.log").open("w") as log:
        process = subprocess.Popen([str(executable), "--settings", str(Path(directory)/"desktop.json"),
                                    "--config", str(Path(directory)/"config.json")], cwd="/", stdout=log, stderr=log)
        try:
            time.sleep(10)
            if process.poll() is not None:
                raise RuntimeError("The normal packaged interface did not stay open; see idle.log")
            cpu, _ = sample(process.pid)
            started = time.monotonic()
            time.sleep(5)
            final_cpu, rss = sample(process.pid)
            wall = time.monotonic() - started
            (output / "idle.json").write_text(json.dumps({
                "status": "PASS", "camera": "off", "warmup_seconds":10,
                "sample_seconds":wall, "cpu_percent_of_one_core":100*(final_cpu-cpu)/wall,
                "rss_mib":rss, "scope":"Idle interface only; live camera resource use remains unverified",
            }, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def prepare_assets():
    fetch(FACE_URL, ROOT / "weights/face_landmarker.task")
    for name, expected in MODELS.items():
        actual = hashlib.sha256((ROOT / "weights" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Bundled model hash mismatch: {name}")
    generated = ROOT / ".cache/macos-build/assets"
    licenses = generated / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    versions = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        versions[name] = distribution.version
        for file in distribution.files or []:
            if any(word in file.name.lower() for word in ("license", "copying", "notice")):
                source = distribution.locate_file(file)
                if source.is_file():
                    destination = licenses / name / str(file).replace("..", "_")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
    (generated / "build-info.json").write_text(json.dumps({
        "models": MODELS, "versions": versions, "architecture": "arm64",
        "minimum_macos": "14.0", "virtual_camera": "pyvirtualcam" in versions,
    }, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Build on an Apple Silicon Mac, using native arm64 Python (no Rosetta).")
    if int(platform.mac_ver()[0].split(".")[0]) < 14 or sys.version_info[:2] != (3,12):
        parser.error("Build with Python 3.12 on macOS 14 or later.")
    prepare_assets()
    import os
    environment = os.environ.copy()
    environment["PYINSTALLER_STRICT_BUNDLE_CODESIGN_ERROR"] = "1"
    output = ROOT / "artifacts/macos"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(output), "--workpath", str(ROOT / ".cache/macos-build/work"),
                    str(ROOT / "packaging/macos.spec")], cwd=ROOT, env=environment, check=True)
    app = output / "Gazeshift.app"
    executable = app / "Contents/MacOS/Gazeshift"
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)
    architectures = subprocess.check_output(["lipo", "-archs", str(executable)], text=True).strip()
    if architectures != "arm64":
        raise ValueError(f"Unexpected executable architecture: {architectures}")
    subprocess.run([str(executable), "--self-test", "--report", str(output / "smoke.json")],
                   cwd="/", check=True, timeout=180)
    check_idle(executable, output)
    subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app),
                    str(output / "Gazeshift-macOS-arm64.zip")], check=True)
    # Include the corresponding tracked source alongside this development build.
    subprocess.run(["git", "archive", "--format=zip", "HEAD", "-o", str(output / "Gazeshift-source.zip")],
                   cwd=ROOT, check=True)
    print(f"Built and checked: {output / 'Gazeshift-macOS-arm64.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
