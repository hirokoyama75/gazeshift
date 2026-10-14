"""Camera-free matched CPU/backend timing, accuracy and execution profiling."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnxruntime as ort
from engine.compositor import EyeCrop
from engine.onnx_warper import OnnxWarper, PROVIDERS, fixed_inputs
from scripts.replay import synthetic_sample


def parity_cases():
    fixture = ROOT / ".cache/parity/dirl_flx12.pkl"
    references = ROOT / ".cache/parity/restored_tf.pkl"
    if fixture.exists() and references.exists():
        data = pickle.loads(fixture.read_bytes())
        truth = pickle.loads(references.read_bytes())
        if (truth["fixture_sha256"] != hashlib.sha256(fixture.read_bytes()).hexdigest()
                or truth["model_sha256"] != hashlib.sha256((ROOT / "weights/flx_12.onnx").read_bytes()).hexdigest()):
            raise ValueError("Parity fixtures are stale; run verify_headless.py --all first")
        if len(data["cases"]) != 8 or len(truth["cases"]) != 8:
            raise ValueError("Incomplete parity fixtures")
        cases = []
        for case, reference in zip(data["cases"], truth["cases"]):
            if case["label"] != reference["label"]:
                raise ValueError("Parity case labels do not match")
            cases.append((case["label"], case["inputs"], reference["outputs"]))
        return "original TensorFlow / real data", cases
    frame, face = synthetic_sample(0)
    inputs = {}
    for side, points in zip(("l", "r"), face.anchors):
        inputs[f"img_{side}"], inputs[f"fp_{side}"] = EyeCrop.from_points(points, face.pose[2]).model_inputs(frame, side)
    cases = []
    for angle in (-15., 0., 15.):
        feed = dict(inputs, ang_l=np.array([[angle, -angle]], np.float32),
                    ang_r=np.array([[angle, -angle]], np.float32))
        cases.append((str(angle), feed, None))
    return "synthetic CPU consistency only; original TensorFlow parity unverified", cases


def benchmark(provider, threads=None):
    engine = OnnxWarper(provider=provider, threads=threads)
    feed = fixed_inputs()
    for _ in range(50):
        engine.session.run(None, feed)
    samples = []
    for _ in range(500):
        start = time.perf_counter_ns()
        engine.session.run(None, feed)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return engine, dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
                       max_ms=max(samples), providers=engine.session.get_providers(),
                       threads=engine.threads, provider_options=engine.session.get_provider_options())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=tuple(PROVIDERS))
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/provider-benchmark.json")
    args = parser.parse_args(argv)
    report = dict(status="FAIL", provider=args.provider, host=platform.platform(),
                  machine=platform.machine(), python=platform.python_version(), onnxruntime=ort.__version__,
                  warmups=50, timed_runs=500, accuracy_limit=1e-4,
                  scope="Eye inference only; live FPS, total latency and camera/OBS remain manual_pending")
    try:
        report["model_sha256"] = hashlib.sha256((ROOT / "weights/flx_12.onnx").read_bytes()).hexdigest()
        cpu, report["cpu"] = benchmark("cpu")
        engine, report["selected"] = (cpu, report["cpu"]) if args.provider == "cpu" else benchmark(args.provider)
        report["parity_reference"], cases = parity_cases()
        errors = {}
        for label, feed, reference in cases:
            expected = reference if reference is not None else cpu.run(feed)
            for name, value in engine.run(feed).items():
                error = float(np.max(np.abs(value - expected[name])))
                errors[f"{label}/{name}"] = error
                if not np.isfinite(error) or error >= 1e-4:
                    raise ValueError(f"Accuracy failed: {label}/{name} = {error}")
        report["errors"] = errors
        profile_dir = args.report.parent / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        # Profiling is separate so instrumentation never affects the timing samples.
        options = engine.session.get_session_options()
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_dir / args.provider)
        session = ort.InferenceSession(str(ROOT / "weights/flx_12.onnx"), sess_options=options,
                                       providers=[(PROVIDERS[args.provider], engine.session.get_provider_options()[PROVIDERS[args.provider]])])
        if PROVIDERS[args.provider] not in session.get_providers():
            raise RuntimeError("Profiling session fell back to CPU")
        session.disable_fallback()
        session.run(None, fixed_inputs())
        profile_path = session.end_profiling()
        nodes = json.loads(Path(profile_path).read_text())
        counts = {}
        for node in nodes:
            name = node.get("args", {}).get("provider")
            if name:
                counts[name] = counts.get(name, 0) + 1
        report["profile"] = dict(path=profile_path, node_counts=counts)
        if not counts.get(PROVIDERS[args.provider]):
            raise RuntimeError("Profile did not confirm execution on the selected backend")
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"{report['status']}: {args.report}")
    if "error" in report:
        print(report["error"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
