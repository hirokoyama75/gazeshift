"""Physical I/O is forbidden in every automated gate, not merely unused."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import pytest

ROOT = Path(__file__).resolve().parents[2]


def pytest_collection_modifyitems(items):
    for item in items:
        if any(item.get_closest_marker(name) is not None for name in ("skip", "skipif", "xfail")):
            raise pytest.UsageError(f"Required headless gates may not skip or xfail: {item.nodeid}")


@pytest.fixture(autouse=True)
def guard_hardware(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Headless gate attempted physical/display I/O")
    real_capture = cv2.VideoCapture
    def file_only(source, *args, **kwargs):
        if not isinstance(source, (str, Path)) or not Path(source).is_file():
            return forbidden()
        return real_capture(str(source), *args, **kwargs)
    for name in ("imshow", "waitKey", "waitKeyEx", "namedWindow"):
        monkeypatch.setattr(cv2, name, forbidden)
    monkeypatch.setattr(cv2, "VideoCapture", file_only)
    import utils.video_io as io
    monkeypatch.setattr(io, "VirtualCamera", forbidden)
    monkeypatch.setattr(io, "discover_virtual_camera", forbidden)
    import utils.monitors
    monkeypatch.setattr(utils.monitors, "get_monitors", forbidden)


@pytest.fixture
def metrics(request):
    data = {}
    yield data
    directory = Path(os.environ.get("GAZESHIFT_GATE_DIR", ROOT / "artifacts/gates"))
    directory.mkdir(parents=True, exist_ok=True)
    filename = request.node.nodeid.replace("/", "_").replace("::", "__").replace(":", "_")
    (directory / (filename + ".json")).write_text(json.dumps(data, indent=2, allow_nan=False))
