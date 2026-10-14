import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest


def test_frozen_mac_cache_outside_bundle(tmp_path, monkeypatch, metrics):
    from utils import paths
    assert paths.cache_root() == paths.ROOT / ".cache"
    with monkeypatch.context() as patch:
        patch.setattr(sys, "frozen", True, raising=False)
        patch.setattr(sys, "platform", "darwin")
        patch.setattr(Path, "home", lambda: tmp_path)
        patch.setattr(paths, "ROOT", tmp_path / "Gazeshift.app/Contents/Resources")
        assert paths.cache_root() == tmp_path / "Library/Caches/Gazeshift"
        assert not paths.cache_root().is_relative_to(paths.ROOT)
    metrics.update(bundle_read_only=True, user_cache_separate=True)


def test_frozen_missing_face_model_never_downloads(tmp_path, monkeypatch, metrics):
    from engine import landmark_tracker as tracker
    from engine.geometry import CameraGeometry
    monkeypatch.setattr(tracker, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace())
    def forbidden(*args): raise AssertionError("Attempted model download into app")
    monkeypatch.setattr(tracker, "fetch", forbidden)
    with pytest.raises(RuntimeError, match="Reinstall Gazeshift.app"):
        tracker.LandmarkTracker(CameraGeometry())
    metrics.update(no_network_when_bundled=True, missing_model_diagnostic=True)


def test_mac_builder_platform_and_model_integrity(tmp_path, monkeypatch, metrics):
    from scripts import build_macos as builder
    monkeypatch.setattr(builder.platform, "system", lambda: "Windows")
    with pytest.raises(SystemExit) as error:
        builder.main([])
    assert error.value.code == 2
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    (tmp_path / "weights").mkdir()
    model = tmp_path / "weights/flx_12.onnx"
    model.write_bytes(b"corrupt model")
    monkeypatch.setattr(builder, "fetch", lambda *args: None)
    with pytest.raises(ValueError, match="hash mismatch"):
        builder.prepare_assets()
    assert model.read_bytes() == b"corrupt model"
    model.write_bytes(b"correct model")
    monkeypatch.setattr(builder, "MODELS", {"flx_12.onnx": hashlib.sha256(model.read_bytes()).hexdigest()})
    monkeypatch.setattr(builder.metadata, "distributions", lambda: [])
    builder.prepare_assets()
    info = json.loads((tmp_path / ".cache/macos-build/assets/build-info.json").read_text())
    assert info["models"] == builder.MODELS and info["architecture"] == "arm64"
    metrics.update(native_build_required=True, model_integrity_enforced=True, build_manifest=True)


def test_bundle_failure_writes_report_without_camera(tmp_path, monkeypatch, metrics):
    from scripts import check_bundle
    monkeypatch.setattr(check_bundle, "ROOT", tmp_path)
    report = tmp_path / "diagnostic.json"
    assert check_bundle.main(["--report", str(report)]) == 1
    result = json.loads(report.read_text())
    assert result["status"] == "FAIL" and "build-info.json" in result["error"]
    assert result["manual_pending"]
    metrics.update(failure_report=True, live_acceptance_not_inferred=True)
