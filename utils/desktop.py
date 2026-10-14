"""Desktop settings and a worker that never touches GUI widgets."""
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import time

from engine.geometry import load_config
from engine.landmark_tracker import LandmarkTracker
from engine.onnx_warper import OnnxWarper, choose_provider
from engine.pipeline import Correction, Pipeline
from utils.calibration import Calibration
from utils.video_io import Capture, VirtualCamera


@dataclass(frozen=True)
class DesktopSettings:
    camera: int = 0
    provider: str = "cpu"
    virtual_camera: bool = False
    mirror: bool = True
    correction: Correction = field(default_factory=Correction)
    theme: str = "dark"

    def validate(self):
        if type(self.camera) is not int or not 0 <= self.camera <= 99:
            raise ValueError("Choose a camera number from 0 to 99")
        if self.provider not in ("auto", "cpu", "coreml", "dml", "cuda"):
            raise ValueError("Unknown processing option")
        if type(self.virtual_camera) is not bool or type(self.mirror) is not bool:
            raise ValueError("Invalid output settings")
        if not isinstance(self.correction, Correction):
            raise ValueError("Invalid correction controls")
        if self.theme not in ("dark", "light"):
            raise ValueError("Choose dark or light appearance")
        return self


def load_settings(path):
    path = Path(path)
    if not path.exists():
        return DesktopSettings()
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Saved controls must be a JSON object")
    data["correction"] = Correction(**data.get("correction", {}))
    return DesktopSettings(**data).validate()


def save_settings(path, settings):
    settings.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".desktop-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(asdict(settings), stream, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def startup_error(stage, error, camera):
    if stage == "camera":
        return (f"Camera {camera} could not be opened. Close OBS and other apps using your webcam, "
                "then try again. Check system camera permissions. On macOS, allow Gazeshift "
                "(Terminal or Python when running from source) in "
                "System Settings > Privacy & Security > Camera, then restart Gazeshift.")
    if stage == "output":
        return ("Call output is off. Install OBS, then leave OBS's own Virtual Camera stopped. "
                "On macOS, start and stop OBS Virtual Camera once and approve its extension. "
                "Select OBS Virtual Camera in your call app. Your preview still works. " + str(error))
    return "Gazeshift could not start: " + str(error)


class DesktopWorker(threading.Thread):
    def __init__(self, settings, config_path, *, capture_factory=Capture,
                 tracker_factory=LandmarkTracker, engine_factory=OnnxWarper,
                 virtual_factory=VirtualCamera, clock=time.monotonic):
        super().__init__(name="gazeshift-desktop", daemon=True)
        self.settings = settings.validate()
        self.config_path, self.clock = config_path, clock
        self.capture_factory, self.tracker_factory = capture_factory, tracker_factory
        self.engine_factory, self.virtual_factory = engine_factory, virtual_factory
        self.commands, self.events = queue.Queue(), queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.stopping = threading.Event()

    def stop(self):
        self.stopping.set()

    def run(self):
        resources, stage = [], "model"
        try:
            provider = self.settings.provider
            if provider == "auto":
                self.events.put(("status", "Comparing processing options…"))
                provider, timings = choose_provider()
                self.events.put(("provider", (provider, timings)))
            else:
                self.events.put(("provider", (provider, {})))
            if self.stopping.is_set(): return
            geometry = load_config(self.config_path)
            geometry = replace(geometry, provider=provider)
            if geometry.reference_width != 1280:
                geometry = replace(geometry, focal_length=geometry.focal_length * 1280/geometry.reference_width,
                                   reference_width=1280)
            engine = self.engine_factory(provider=provider)
            if self.stopping.is_set(): return
            tracker = self.tracker_factory(geometry)
            resources.append(tracker)
            if self.stopping.is_set(): return
            stage = "camera"
            capture = self.capture_factory(self.settings.camera, 1280, 720)
            resources.append(capture)
            if self.stopping.is_set(): return
            first_sample = capture.read()
            if first_sample is None:
                raise RuntimeError("Camera opened but returned no frames")
            height, width = first_sample[0].shape[:2]
            if width != geometry.reference_width:
                geometry = replace(geometry, focal_length=geometry.focal_length * width/geometry.reference_width,
                                   reference_width=width)
                tracker.geometry = geometry
            virtual = None
            if self.settings.virtual_camera:
                try:
                    virtual = self.virtual_factory(width, height)
                    resources.append(virtual)
                except (ImportError, RuntimeError) as error:
                    self.events.put(("warning", startup_error("output", error, self.settings.camera)))
            pipeline = Pipeline(engine, geometry, self.settings.correction)
            calibration = Calibration(geometry, self.config_path, clock=self.clock)
            self.events.put(("started", bool(virtual)))
            count, started, last_state = 0, self.clock(), "idle"
            stage = "processing"
            while not self.stopping.is_set():
                while True:
                    try:
                        action, value = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if action == "correction": pipeline.correction = value
                    elif action == "enabled": pipeline.enabled = value
                    elif action == "calibrate": calibration.start()
                    elif action == "cancel": calibration.cancel()
                sample = first_sample if first_sample is not None else capture.read()
                first_sample = None
                if sample is None: break
                frame, timestamp = sample
                tracker.calibrating = calibration.active
                face = tracker.track(frame, timestamp)
                candidate = calibration.update(face.observation if face else None)
                if candidate:
                    tracker.geometry = pipeline.geometry = calibration.geometry = candidate
                if calibration.state != last_state:
                    last_state = calibration.state
                    self.events.put(("calibration", (last_state, calibration.message)))
                result = pipeline.process(frame, face)
                if virtual: virtual.send(result)
                count += 1
                packet = (frame, result, face is not None, count/max(.001, self.clock()-started),
                          len(calibration.lens), len(calibration.screen))
                try: self.frames.get_nowait()
                except queue.Empty: pass
                self.frames.put_nowait(packet)
        except Exception as error:
            if not self.stopping.is_set():
                self.events.put(("error", startup_error(stage, error, self.settings.camera)))
        finally:
            for resource in reversed(resources):
                try: resource.close()
                except Exception as error: self.events.put(("error", "Could not release a device: " + str(error)))
            self.events.put(("stopped", None))
