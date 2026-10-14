"""Measure CPU thread choices without opening a camera or changing settings."""
import argparse
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.benchmark_provider import benchmark
from engine.onnx_warper import fixed_inputs


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
    args.report.write_text(json.dumps({"host": platform.platform(), "machine": platform.machine(),
                                      "warmups":50, "timed_runs":500, "results":results}, indent=2))


if __name__ == "__main__":
    main()
