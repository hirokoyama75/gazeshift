import json
import os
import subprocess
import sys
import time
import pickle
import numpy as np
import onnx
import pytest
from engine.onnx_warper import OnnxWarper
from engine.compositor import remap_patch
from scripts.assets import ROOT, sha256, git_blob
from scripts.sampler_reference import tf_sampler, coordinates


def test_export_schema_and_provenance(metrics):
    export = json.loads((ROOT / "artifacts/export.json").read_text())
    assert export["python"].startswith("3.11.")
    assert export["tensorflow"] == "2.15.0" and export["tf2onnx"] == "1.17.0"
    assert export["restored_variables"] == {"L": 140, "R": 140}
    assert export["model_sha256"] == sha256(ROOT / "weights/flx_12.onnx")
    manifest = json.loads((ROOT / "scripts/source_manifest.json").read_text())
    assert len(manifest["assets"]) >= 10
    for entry in manifest["assets"]:
        path = ROOT / entry["local"]
        assert git_blob(path.read_bytes()) == entry["git_blob"]
        assert sha256(path) == export["hashes"][entry["local"]]
    model = onnx.load(ROOT / "weights/flx_12.onnx")
    onnx.checker.check_model(model)
    assert next(o.version for o in model.opset_import if o.domain == "") == 15
    assert not any("apply_transformation" in n.name or n.op_type == "Floor" for n in model.graph.node)
    assert not any(n.op_type in ("Shape", "Gather", "Slice") for n in model.graph.node)
    assert [i.name for i in model.graph.initializer] == sorted({i.name for i in model.graph.initializer})
    assert all(i.name.startswith('constant_') for i in model.graph.initializer)
    softmax = [n for n in model.graph.node if n.op_type == "Softmax"]
    assert len(softmax) == 2 and all(next(a.i for a in n.attribute if a.name == 'axis') == 3 for n in softmax)
    OnnxWarper()
    assert (ROOT / ".cache/parity/DATASET_NOTICE.txt").stat().st_size > 0
    metrics.update(export)


def test_real_tensor_and_sampler_parity(metrics):
    fixture_path = ROOT / ".cache/parity/dirl_flx12.pkl"
    references = pickle.loads((ROOT / ".cache/parity/restored_tf.pkl").read_bytes())
    fixture = pickle.loads(fixture_path.read_bytes())
    assert references["fixture_sha256"] == sha256(fixture_path)
    assert references["model_sha256"] == sha256(ROOT / "weights/flx_12.onnx")
    assert fixture["schema"] == 1 and fixture["seed"] == 0
    assert len(fixture["cases"]) == len(references["cases"]) == 8
    engine = OnnxWarper()
    results = []
    for case, reference in zip(fixture["cases"], references["cases"]):
        assert case["label"] == reference["label"]
        actual = engine.run(case["inputs"])
        errors = {}
        for key, value in actual.items():
            error = float(np.max(np.abs(value - reference["outputs"][key])))
            errors[key] = error
            assert error < 1e-4, (case["label"], key, error)
        for side in ("l", "r"):
            flow = reference["outputs"][f"flow_{side}"][0]
            image = case["inputs"][f"img_{side}"][0]
            expected = reference["outputs"][f"sampler_{side}"][0]
            faithful = tf_sampler(image, flow)
            error = float(np.max(np.abs(faithful - expected)))
            assert error < 2/255, error
            x, y = coordinates(flow)
            interior = (x >= 0) & (x < 63) & (y >= 0) & (y < 47)
            assert interior.any()
            runtime = remap_patch(image, flow)
            delta = np.abs(runtime - expected)
            interior_error = float(delta[interior].max())
            assert interior_error < 2/255, (side, interior_error)
            errors[f"{side}_sampler_full"] = error
            errors[f"{side}_interior_error"] = interior_error
            errors[f"{side}_interior_coverage"] = float(interior.mean())
            errors[f"{side}_boundary_error"] = float(delta[~interior].max()) if (~interior).any() else 0
        results.append({"case": case["label"], **errors})
    metrics["cases"] = results
    metrics["inference_calls"] = engine.calls


def test_finite_synthetic_replay(metrics):
    from scripts.replay import synthetic_replay
    result = synthetic_replay()
    assert result["frames"] == 90
    assert result["inference_calls"] > 0
    assert result["label"] == "SYNTHETIC"
    assert np.isfinite(result["synthetic_iris_center_rms_px"])
    for path in result["artifacts"]:
        assert (ROOT / path).stat().st_size > 0
        import cv2
        decoder = cv2.VideoCapture(str(ROOT / path))
        count = 0
        try:
            for _ in range(91):
                ok, frame = decoder.read()
                if not ok:
                    break
                assert frame.shape[0] == 720
                count += 1
        finally:
            decoder.release()
        assert count == 90
    metrics.update(result)


def test_verifier_failure_reports_and_deadline(tmp_path,metrics):
    from scripts.verify_headless import read_results,run_child,StageError
    empty=tmp_path/'empty.xml'
    empty.write_text('<testsuites><testsuite tests="0"/></testsuites>')
    with pytest.raises(ValueError,match='Empty or incomplete'):
        read_results(empty,'m0',tmp_path)
    started=time.monotonic()
    with pytest.raises(StageError) as deadline:
        run_child([sys.executable,'-c','import time;time.sleep(30)'],'deadline_probe',started+.2,tmp_path,os.environ.copy())
    elapsed=time.monotonic()-started
    assert deadline.value.status=='TIMEOUT' and elapsed<2
    reports={}
    for case,timeout,expected,code in [('timeout','0','TIMEOUT',3),('dependency','2','UNAVAILABLE',2)]:
        report=tmp_path/f'{case}.json'
        environment=os.environ.copy()
        if case=='dependency':environment['PATH']=''
        result=subprocess.run([sys.executable,'scripts/verify_headless.py','--milestone','m3','--timeout-seconds',timeout,'--report',str(report)],
                              cwd=ROOT,env=environment,capture_output=True,timeout=5)
        parsed=json.loads(report.read_text())
        assert result.returncode==code and parsed['status']==expected
        assert parsed['milestones']['m3']['status']=='NOT_RUN'
        assert all(item['status']=='manual_pending' for item in parsed['manual_pending'])
        reports[case]=dict(exit_code=result.returncode,status=parsed['status'])
    metrics.update(empty_suite_rejected=True,deadline_seconds=elapsed,failure_reports=reports)
