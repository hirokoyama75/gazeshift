"""Bounded, hardware-free M0–M3 orchestration. Always writes a diagnostic JSON."""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.assets import sha256

REQUIRED = {
    "m0": ["test_export_schema_and_provenance", "test_real_tensor_and_sampler_parity", "test_finite_synthetic_replay", "test_verifier_failure_reports_and_deadline"],
    "m1": ["test_zero_strength_exact_and_inference_bypass", "test_complete_coordinate_fade_and_rolled_fractional_crops",
           "test_production_against_real_tf_interior", "test_rolled_sampling_map_against_independent_frame_bilinear", "test_one_euro_stationary_variance", "test_blink_shared_min_immediate_close_linear_reopen"],
    "m2": ["test_real_cpu_benchmark", "test_dshow_order_latest_frame_and_bounded_shutdown", "test_injected_720p_pipeline_and_cleanup", "test_tracker_rgb_raw_blink_and_calibration_only_iris", "test_projected_head_pose_matches_screen_facing_direction", "test_cli_calibration_keys_persistence_and_clean_output"],
    "m3": ["test_independent_two_look_recovery", "test_failures_preserve_config_bytes", "test_three_degree_wrapped_boundary_and_secondary_monitor", "test_screen_relative_pose_cone"],
}
MIN_COUNTS = {"m0": 4, "m1": 6, "m2": 10, "m3": 24}
MANUAL = ["Windows and macOS physical camera / OBS selection", "720p sustained >=30 fps",
          "capture-to-output p95 <33 ms", "visual eye / blink / seam quality", "real-subject Calibration accuracy", "RSS after 60 seconds"]


class StageError(Exception):
    def __init__(self, stage, status, detail):
        self.stage, self.status, self.detail = stage, status, detail
        super().__init__(f"{stage}: {detail}")


def start_windows_job(process):
    """Own a suspended child's entire process tree before letting it execute."""
    from ctypes import wintypes
    class Limits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                    ("active", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                    ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
    class Counters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
    class Extended(ctypes.Structure):
        _fields_ = [("basic", Limits), ("io", Counters), ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())
        native = ctypes.WinDLL("ntdll")
        native.NtResumeProcess.argtypes = [wintypes.HANDLE]
        native.NtResumeProcess.restype = ctypes.c_long
        if native.NtResumeProcess(wintypes.HANDLE(int(process._handle))) < 0:
            raise OSError("Cannot resume bounded child process")
        return kernel, handle
    except BaseException:
        kernel.CloseHandle(handle)
        process.kill()
        process.wait(timeout=5)
        raise


def terminate_tree(process, job=None):
    if os.name == "nt":
        if job:
            job[0].TerminateJobObject(job[1], 3)
        else:
            process.kill()
    else:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    process.wait(timeout=5)


def run_child(command, stage, deadline, directory, env, limit=300, unavailable=False):
    remaining = min(limit, deadline-time.monotonic())
    if remaining <= 0:
        raise StageError(stage, "TIMEOUT", "Overall deadline expired")
    log = directory / f"{stage}.log"
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | 0x4} if os.name == "nt" else {"start_new_session": True}
    job = None
    process = None
    try:
        with log.open("wb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
            if os.name == "nt":
                job = start_windows_job(process)
            remaining = min(remaining, deadline-time.monotonic())
            if remaining <= 0:
                terminate_tree(process, job)
                raise StageError(stage, "TIMEOUT", "Overall deadline expired during child setup")
            try:
                code = process.wait(timeout=remaining)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                terminate_tree(process, job)
                raise StageError(stage, "TIMEOUT", f"Child exceeded {remaining:.1f} seconds; process tree terminated; see {log}")
    except OSError as error:
        raise StageError(stage, "UNAVAILABLE", str(error)) from error
    finally:
        if job:
            job[0].CloseHandle(job[1])
        elif process is not None and os.name != "nt" and process.poll() is not None:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
    if code:
        detail = log.read_text(errors="replace")[-10000:]
        raise StageError(stage, "UNAVAILABLE" if unavailable else "FAIL", f"exit {code}: {detail}")
    return {"exit_code": code, "log": str(log.relative_to(ROOT))}


def read_results(path, milestone, gate_directory):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if len(cases) < MIN_COUNTS[milestone]:
        raise ValueError(f"Empty or incomplete suite: {len(cases)} cases for {milestone}")
    names = {case.get("name").split("[")[0] for case in cases}
    if set(REQUIRED[milestone]) - names:
        raise ValueError(f"Missing required gates: {set(REQUIRED[milestone])-names}")
    counts = {"collected": len(cases), "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfail": 0}
    gates = []
    for name in REQUIRED[milestone]:
        selected = [case for case in cases if case.get("name").split("[")[0] == name]
        records = []
        for case in selected:
            failed = case.find("failure") is not None or case.find("error") is not None
            skipped = case.find("skipped") is not None
            counts["xfail"] += int(skipped and case.find("skipped").get("type") == "pytest.xfail")
            counts["failed"] += int(case.find("failure") is not None)
            counts["errors"] += int(case.find("error") is not None)
            counts["skipped"] += int(skipped)
            counts["passed"] += int(not failed and not skipped)
            metric_file = next((p for p in gate_directory.iterdir() if p.name.endswith(f"__{case.get('name')}.json")), None)
            if metric_file is None:
                raise ValueError(f"Missing current-run metrics for {case.get('name')}")
            records.append({"case": case.get("name"), "status": "FAIL" if failed or skipped else "PASS", "metrics": json.loads(metric_file.read_text())})
        gates.append({"name": name, "status": "PASS" if all(r["status"] == "PASS" for r in records) else "FAIL", "test_count": len(selected), "cases": records})
    status = "PASS" if counts["passed"] == len(cases) and not counts["skipped"] else "FAIL"
    return {"status": status, "counts": counts, "gates": gates}


def main(argv=None):
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--milestone", choices=REQUIRED)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/verification.json")
    args = parser.parse_args(argv)
    start = time.monotonic()
    deadline = start + args.timeout_seconds
    selected = list(REQUIRED) if args.all else [args.milestone]
    run_id = uuid.uuid4().hex
    directory = ROOT / "artifacts/runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    gate_directory = directory / "metrics"
    gate_directory.mkdir()
    env = os.environ.copy()
    env.update(UV_CACHE_DIR=str(ROOT / ".cache/uv"), UV_PYTHON_INSTALL_DIR=str(ROOT / ".cache/python"),
               GAZESHIFT_GATE_DIR=str(gate_directory), PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    report = {"schema": 1, "run_id": run_id, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "FAIL", "scope": "all" if args.all else args.milestone, "seed": 0,
              "runtime": {"python": platform.python_version(), "executable": sys.executable, "host": platform.platform(), "cpu": platform.processor()},
              "milestones": {m: {"status": "NOT_RUN", "counts": {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfail": 0},
                  "gates": [{"name": name, "status": "NOT_RUN", "test_count": 0, "cases": []} for name in REQUIRED[m]]} for m in selected}, "stages": {},
              "manual_pending": [{"check": item, "status": "manual_pending"} for item in MANUAL], "diagnostics": []}
    stage = "runtime"
    try:
        if sys.version_info[:2] != (3, 12):
            raise StageError(stage, "UNAVAILABLE", "Runtime must be Python 3.12")
        uv = shutil.which("uv")
        if not uv: raise StageError("setup", "UNAVAILABLE", "uv is not on PATH")
        if args.all or args.milestone == "m0":
            commands = [
                ("assets", [sys.executable, "scripts/assets.py"], True),
                ("exporter_setup", [uv, "venv", "--allow-existing", "--python", "3.11", str(ROOT / ".cache/exporter")], True),
            ]
            for stage, command, unavailable in commands:
                report["stages"][stage] = run_child(command, stage, deadline, directory, env, unavailable=unavailable)
            exporter = ROOT / (".cache/exporter/Scripts/python.exe" if os.name == "nt" else ".cache/exporter/bin/python")
            stage = "exporter_dependencies"
            report["stages"][stage] = run_child([uv, "pip", "install", "--python", str(exporter), "-r", "scripts/export-requirements.txt"], stage, deadline, directory, env, unavailable=True)
            stage = "export"
            # Always restore and export anew, including fresh TF reference outputs.
            report["stages"][stage] = run_child([str(exporter), "scripts/export_tf_to_onnx.py"], stage, deadline, directory, env)
            report["export"] = json.loads((ROOT / "artifacts/export.json").read_text())
        gate_errors = []
        for milestone in selected:
            stage = milestone
            junit = directory / f"{milestone}.xml"
            command = [sys.executable, "-m", "pytest", f"tests/headless/test_{milestone}.py", "-q", "--basetemp", str(directory / f"tmp_{milestone}"), "--junitxml", str(junit)]
            child_error = None
            try:
                report["stages"][stage] = run_child(command, stage, deadline, directory, env)
            except StageError as error:
                child_error = error
            if junit.exists():
                report["milestones"][milestone] = read_results(junit, milestone, gate_directory)
            if child_error:
                gate_errors.append(child_error)
                if child_error.status == "TIMEOUT":
                    raise child_error
                continue
            if report["milestones"][milestone]["status"] != "PASS":
                gate_errors.append(StageError(stage, "FAIL", "A required gate failed/skipped or the suite was incomplete"))
        if gate_errors:
            for error in gate_errors[1:]:
                report["diagnostics"].append({"stage": error.stage, "detail": error.detail})
            raise gate_errors[0]
        artifacts = ["weights/flx_12.onnx", ".cache/parity/dirl_flx12.pkl", ".cache/parity/restored_tf.pkl"]
        if "m0" in selected:
            artifacts += ["artifacts/replay/synthetic_raw.avi", "artifacts/replay/synthetic_corrected.avi", "artifacts/replay/synthetic_raw_corrected.avi", "artifacts/replay/metrics.json"]
        report["artifacts"] = {}
        for item in artifacts:
            path = ROOT / item
            if not path.is_file() or path.stat().st_size == 0:
                raise StageError("artifacts", "FAIL", f"Missing/empty artifact: {item}")
            report["artifacts"][item] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        report["status"] = "PASS"
    except StageError as error:
        report["status"] = error.status
        report["diagnostics"].append({"stage": error.stage, "detail": error.detail})
    except BaseException as error:
        report["status"] = "FAIL"
        report["diagnostics"].append({"stage": stage, "detail": str(error), "traceback": traceback.format_exc()})
    finally:
        # Failure reports also describe existing files, without treating stale
        # artifacts as evidence that an unexecuted gate passed.
        report["observed_artifacts"] = {}
        for name in ("weights/flx_12.onnx", ".cache/parity/dirl_flx12.pkl", ".cache/parity/restored_tf.pkl",
                     "weights/face_landmarker.task", "artifacts/replay/synthetic_raw.avi", "artifacts/replay/synthetic_corrected.avi",
                     "artifacts/replay/synthetic_raw_corrected.avi", "artifacts/replay/metrics.json"):
            path = ROOT / name
            if path.is_file():
                report["observed_artifacts"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        report["elapsed_seconds"] = time.monotonic()-start
        report["report"] = str(args.report.resolve())
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({"status": report["status"], "report": report["report"], "elapsed_seconds": report["elapsed_seconds"],
                          "milestones": {k: v["status"] for k,v in report["milestones"].items()}, "diagnostics": report["diagnostics"]}, indent=2))
    return {"PASS": 0, "FAIL": 1, "UNAVAILABLE": 2, "TIMEOUT": 3}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
