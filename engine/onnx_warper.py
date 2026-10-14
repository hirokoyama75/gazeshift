"""One both-eye ONNX session; provider probing never installs runtime wheels."""
from pathlib import Path
import hashlib
import os
import sys
import time
import numpy as np
import onnxruntime as ort
from utils.paths import ROOT, cache_root

PROVIDERS = {"cpu": "CPUExecutionProvider", "dml": "DmlExecutionProvider",
             "cuda": "CUDAExecutionProvider", "coreml": "CoreMLExecutionProvider"}


def fixed_inputs():
    rng = np.random.default_rng(0)
    values = {}
    for side in ("l", "r"):
        values[f"img_{side}"] = rng.random((1, 48, 64, 3), dtype=np.float32)
        values[f"fp_{side}"] = rng.uniform(-32, 32, (1, 48, 64, 12)).astype(np.float32)
        values[f"ang_{side}"] = np.array([[12., -7.]], np.float32)
    return values


class OnnxWarper:
    def __init__(self, model=ROOT / "weights/flx_12.onnx", provider="cpu", threads=None):
        threads = min(6, os.cpu_count() or 1) if threads is None else threads
        name = PROVIDERS[provider]
        if name not in ort.get_available_providers():
            raise RuntimeError(f"{name} not installed; use matching uv selectors")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if sys.platform == "darwin":
            # Sleep between camera frames while retaining fast scheduling within each run.
            options.add_session_config_entry("session.force_spinning_stop", "1")
        if provider == "dml":
            options.enable_mem_pattern = False
        provider_options = {}
        if provider == "cuda":
            ort.preload_dlls()
            # Keep the original fp32 accuracy; CUDA enables reduced-precision TF32 by default.
            provider_options = {"use_tf32": "0", "cudnn_conv_algo_search": "DEFAULT"}
        elif provider == "coreml":
            digest = hashlib.sha256(Path(model).read_bytes()).hexdigest()
            provider_options = {
                "ModelFormat": "NeuralNetwork", "MLComputeUnits": "CPUAndGPU",
                "RequireStaticInputShapes": "1", "AllowLowPrecisionAccumulationOnGPU": "0",
                "ModelCacheDirectory": str(cache_root() / "coreml" / ort.__version__ / digest),
            }
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=[(name, provider_options)])
        if name not in self.session.get_providers():
            raise RuntimeError(f"{name} could not start; CPU fallback was rejected. "
                               "For NVIDIA, update your driver and use Start Gazeshift NVIDIA GPU.bat. "
                               "Otherwise select CPU or Automatic in Processing.")
        self.session.disable_fallback()
        self.provider_options = provider_options
        self.provider, self.threads, self.calls = provider, threads, 0
        expected = {f"{kind}_{side}": shape for side in ("l", "r") for kind, shape in
                    (("img", [1, 48, 64, 3]), ("fp", [1, 48, 64, 12]), ("ang", [1, 2]))}
        if {v.name: v.shape for v in self.session.get_inputs()} != expected:
            raise ValueError("Unexpected FLX input schema")
        if {v.name: v.shape for v in self.session.get_outputs()} != {f"{k}_{s}": [1, 48, 64, 2] for s in ("l", "r") for k in ("flow", "lcm")}:
            raise ValueError("Unexpected FLX output schema")

    def run(self, inputs):
        for name, value in inputs.items():
            if value.dtype != np.float32 or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid tensor {name}")
        self.calls += 1
        return dict(zip([v.name for v in self.session.get_outputs()], self.session.run(None, inputs)))


def choose_provider(model=ROOT / "weights/flx_12.onnx"):
    available = ort.get_available_providers()
    timings = {}
    inputs = fixed_inputs()
    for provider in ("coreml", "dml", "cuda", "cpu"):
        if PROVIDERS[provider] not in available:
            continue
        try:
            engine = OnnxWarper(model, provider)
            for _ in range(10):
                engine.session.run(None, inputs)
            samples = []
            for _ in range(20):
                start = time.perf_counter_ns()
                engine.session.run(None, inputs)
                samples.append(time.perf_counter_ns() - start)
            timings[provider] = float(np.percentile(samples, 95))
        except Exception as error:
            print(f"Provider {provider} unavailable: {error}")
    if not timings:
        raise RuntimeError("No usable ONNX execution provider")
    return min(timings, key=timings.get), timings
