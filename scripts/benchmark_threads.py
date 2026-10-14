"""Measure CPU thread choices without opening a camera or changing settings."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnxruntime as ort
from scripts.benchmark_provider import benchmark
from engine.onnx_warper import OnnxWarper, fixed_inputs, ROOT


def paced_inference(spinning, threads):
    engine = OnnxWarper(threads=threads)
    options = engine.session.get_session_options()
    options.add_session_config_entry("session.intra_op.allow_spinning", str(spinning))
    engine.session = ort.InferenceSession(str(ROOT / "weights/flx_12.onnx"), sess_options=options,
                                         providers=["CPUExecutionProvider"])
    feed = fixed_inputs()
    for _ in range(50): engine.run(feed)
    samples = []
    started, cpu = time.monotonic(), time.process_time()
    for index in range(90):
        tick = time.perf_counter_ns()
        engine.run(feed)
        samples.append((time.perf_counter_ns()-tick)/1e6)
        time.sleep(max(0, started+(index+1)/30-time.monotonic()))
    elapsed, used = time.monotonic()-started, time.process_time()-cpu
    return dict(cpu_percent_of_one_core=100*used/elapsed, p95_ms=float(np.percentile(samples,95)),
                delivered_fps=90/elapsed, frames=90, threads=threads, spinning=spinning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    results, reference = {}, None
    for threads in (1,2,3,4,6):
        engine, result = benchmark("cpu", threads=threads)
        values = engine.run(fixed_inputs())
        if reference is None:
            reference = values
        error = max(float(np.max(np.abs(values[key] - reference[key]))) for key in values)
        if not np.isfinite(error) or error >= 1e-4:
            raise ValueError(f"Thread choice changed model accuracy: {threads}: {error}")
        result["max_error"] = error
        results[str(threads)] = result
        del engine
    args.report.parent.mkdir(parents=True, exist_ok=True)
    threads = min((int(value) for value in results), key=lambda n: results[str(n)]["p95_ms"])
    paced = {str(spinning): paced_inference(spinning, threads) for spinning in (1,0)}
    args.report.write_text(json.dumps({"host": platform.platform(), "machine": platform.machine(),
                                      "warmups":50, "timed_runs":500, "results":results,
                                      "paced_synthetic_inference":paced}, indent=2))


if __name__ == "__main__":
    main()
