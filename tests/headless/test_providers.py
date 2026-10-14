from types import SimpleNamespace
import ctypes
import json
import pytest
from engine import onnx_warper as module


def fake_session(monkeypatch, active=None):
    sessions = []
    class Session:
        def __init__(self, path, sess_options, providers):
            self.options, self.providers = sess_options, providers
            self.disabled = False
            sessions.append(self)
        def get_providers(self):
            return active if active is not None else [self.providers[0][0], "CPUExecutionProvider"]
        def disable_fallback(self): self.disabled = True
        def get_inputs(self):
            return [SimpleNamespace(name=f"{kind}_{side}", shape=shape)
                    for side in ("l", "r") for kind, shape in
                    (("img", [1,48,64,3]), ("fp", [1,48,64,12]), ("ang", [1,2]))]
        def get_outputs(self):
            return [SimpleNamespace(name=f"{kind}_{side}", shape=[1,48,64,2])
                    for side in ("l", "r") for kind in ("flow", "lcm")]
    monkeypatch.setattr(module.ort, "InferenceSession", Session)
    monkeypatch.setattr(module.ort, "get_available_providers", lambda: list(module.PROVIDERS.values()))
    return sessions


def test_cuda_full_precision_and_fallback_rejection(monkeypatch, metrics):
    preloads = []
    monkeypatch.setattr(module.ort, "preload_dlls", lambda: preloads.append(True))
    sessions = fake_session(monkeypatch)
    module.OnnxWarper(provider="cuda")
    assert preloads == [True] and sessions[0].disabled
    assert sessions[0].providers == [("CUDAExecutionProvider", {"use_tf32": "0", "cudnn_conv_algo_search": "DEFAULT"})]
    fake_session(monkeypatch, active=["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CPU fallback was rejected"):
        module.OnnxWarper(provider="cuda")
    metrics.update(cuda_preloaded=True, tf32_disabled=True, silent_fallback_rejected=True)


def test_coreml_static_precision_and_content_cache(tmp_path, monkeypatch, metrics):
    sessions = fake_session(monkeypatch)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"first version")
    module.OnnxWarper(model, provider="coreml")
    options = sessions[-1].providers[0][1]
    assert options["RequireStaticInputShapes"] == "1"
    assert options["ModelFormat"] == "NeuralNetwork" and options["MLComputeUnits"] == "CPUAndGPU"
    assert options["AllowLowPrecisionAccumulationOnGPU"] == "0"
    model.write_bytes(b"changed weights")
    module.OnnxWarper(model, provider="coreml")
    assert options["ModelCacheDirectory"] != sessions[-1].providers[0][1]["ModelCacheDirectory"]
    metrics.update(static_shapes=True, gpu_fp32_accumulation=True, cache_changes_with_model=True)


def test_provider_probe_warmup_and_cpu_fallback(monkeypatch, metrics):
    clock, calls = [0], {}
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: clock[0])
    monkeypatch.setattr(module.ort, "get_available_providers", lambda: list(module.PROVIDERS.values()))
    class Engine:
        def __init__(self, model, provider):
            if provider == "cuda": raise RuntimeError("driver missing")
            self.session, self.provider = self, provider
            calls[provider] = 0
        def run(self, *args):
            calls[self.provider] += 1
            # Compilation/first runs must not affect the selection.
            clock[0] += 10000 if calls[self.provider] <= 10 else {"cpu":10,"coreml":5,"dml":20}[self.provider]
    monkeypatch.setattr(module, "OnnxWarper", Engine)
    provider, timings = module.choose_provider()
    assert provider == "coreml" and timings == {"coreml":5,"dml":20,"cpu":10}
    assert calls == {"coreml":30,"dml":30,"cpu":30}
    monkeypatch.setattr(module.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert module.choose_provider()[0] == "cpu"
    monkeypatch.setattr(module.ort, "get_available_providers", lambda: [])
    with pytest.raises(RuntimeError, match="No usable"):
        module.choose_provider()
    metrics.update(warmups=10, timed_runs=20, fastest_selected=True, failed_backend_excluded=True)


def test_desktop_auto_uses_resolved_provider(tmp_path, monkeypatch, metrics):
    from utils import desktop
    providers, geometries = [], []
    monkeypatch.setattr(desktop, "choose_provider", lambda: ("coreml", {"coreml":123.}))
    class Tracker:
        def __init__(self, geometry): geometries.append(geometry)
        def close(self): pass
    class Capture:
        def __init__(self, *args): pass
        def read(self): return None
        def close(self): pass
    worker = desktop.DesktopWorker(desktop.DesktopSettings(provider="auto"), tmp_path / "config.json",
                                   engine_factory=lambda **kw: providers.append(kw["provider"]),
                                   tracker_factory=Tracker, capture_factory=Capture)
    worker.start(); worker.join(2)
    assert not worker.is_alive() and providers == ["coreml"]
    assert geometries[0].provider == "coreml"
    assert ("provider", ("coreml", {"coreml":123.})) in list(worker.events.queue)
    assert not (tmp_path / "config.json").exists()
    metrics.update(actual_provider_reported=True, geometry_provider_resolved=True, settings_untouched=True)


def test_mac_camera_names_match_opencv_indices(monkeypatch, metrics):
    from utils import cameras
    selectors, drained, media = {}, [], []
    class Function:
        def __init__(self, function): self.function = function
        def __call__(self, *args): return self.function(*args)
    def selector(value):
        index = len(selectors) + 1
        selectors[index] = value
        return index
    objc = SimpleNamespace(objc_getClass=Function(lambda name: name),
                           sel_registerName=Function(selector), objc_msgSend=ctypes.c_void_p(1))
    def dispatch(receiver, sel, *args):
        name = selectors[sel]
        if name == b"new": return "pool"
        if name == b"drain": drained.append(receiver); return
        if name == b"stringWithUTF8String:": return args[0]
        if name == b"devicesWithMediaType:":
            media.append(args[0])
            return ["FaceTime HD", "OBS Virtual Camera"] if args[0] == b"vide" else ["外部カメラ"]
        if name == b"arrayByAddingObjectsFromArray:": return receiver + args[0]
        if name == b"count": return len(receiver)
        if name == b"objectAtIndex:": return receiver[args[0]]
        if name == b"localizedName": return receiver
        if name == b"UTF8String": return receiver.encode("utf-8")
        raise AssertionError(name)
    monkeypatch.setattr(cameras.ctypes, "CDLL", lambda path: objc)
    monkeypatch.setattr(cameras.ctypes, "CFUNCTYPE", lambda *args: lambda address: dispatch)
    assert cameras.mac_cameras() == [(0,"FaceTime HD"),(2,"外部カメラ")]
    assert media == [b"vide", b"muxx"] and drained == ["pool"]
    metrics.update(native_names=True, video_then_muxed=True, obs_excluded_without_reindexing=True)


def test_avfoundation_capture_and_cleanup(metrics):
    import cv2
    from utils.video_io import Capture
    calls = []
    class Device:
        def isOpened(self): return True
        def set(self, prop, value): calls.append((prop, value))
        def read(self): return False, None
        def release(self): calls.append("released")
    def factory(index, backend):
        assert (index, backend) == (2, cv2.CAP_AVFOUNDATION)
        return Device()
    capture = Capture(2, factory=factory, system="Darwin")
    capture.close()
    assert not capture.thread.is_alive()
    assert (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) not in calls
    assert calls[:3] == [(cv2.CAP_PROP_FRAME_WIDTH,1280),(cv2.CAP_PROP_FRAME_HEIGHT,720),(cv2.CAP_PROP_FPS,30)]
    assert "released" in calls
    metrics.update(backend="AVFoundation", requested_resolution=[1280,720], fps=30, cleanup=True)


def test_provider_benchmark_reference_and_failure_report(tmp_path, monkeypatch, metrics):
    from scripts import benchmark_provider as script
    monkeypatch.setattr(script, "ROOT", tmp_path)
    label, cases = script.parity_cases()
    assert "unverified" in label and len(cases) == 3 and all(expected is None for _, _, expected in cases)
    assert cases[0][1]["img_l"].shape == (1,48,64,3)
    report = tmp_path / "report.json"
    assert script.main(["--provider","coreml","--report",str(report)]) == 1
    data = json.loads(report.read_text())
    assert data["status"] == "FAIL" and "error" in data and data["accuracy_limit"] == 1e-4
    metrics.update(fresh_clone_reference_labeled=True, diagnostic_report_on_failure=True, accuracy_limit=1e-4)
