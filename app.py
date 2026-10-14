"""gazeshift v2.0 CLI. Run with matching uv group/extra selectors."""
import os
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import argparse
from dataclasses import replace
from pathlib import Path
import time
from engine.geometry import load_config, save_config, nudge
from engine.onnx_warper import OnnxWarper, choose_provider
from engine.landmark_tracker import LandmarkTracker
from engine.pipeline import Pipeline
from utils.calibration import Calibration
from utils.video_io import Capture, VirtualCamera, discover_virtual_camera, Preview


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Apply a geometric Gaze Offset with FLX")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--virtual-cam", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--provider", choices=("cpu", "coreml", "dml", "cuda"))
    parser.add_argument("--mirror-preview", action="store_true")
    parser.add_argument("--mirror-capture", action="store_true")
    parser.add_argument("--config", type=Path, default=Path.home() / ".gaze_cam/config.json")
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error("Resolution must be positive")
    return args


def main(argv=None, clock=time.monotonic):
    args = arguments(argv)
    geometry = load_config(args.config)
    if args.width != geometry.reference_width:
        geometry = replace(geometry, focal_length=geometry.focal_length * args.width/geometry.reference_width, reference_width=args.width)
    provider = args.provider
    if provider is None:
        provider, timings = choose_provider()
        print(f"Provider probe (20 runs each, ns): {timings}")
        # Persist provider without changing the resolution-independent focal length.
        save_config(args.config, replace(load_config(args.config), provider=provider))
        geometry = replace(geometry, provider=provider)
    print(f"ONNX provider: {provider}")
    if not args.preview:
        print("Calibration and keyboard controls need --preview. Ctrl+C quits.")
    engine = OnnxWarper(provider=provider)
    pipeline = Pipeline(engine, geometry)
    calibration = Calibration(geometry, args.config, clock=clock)
    resources = []
    overlay, count, started = False, 0, clock()
    try:
        tracker = LandmarkTracker(geometry)
        resources.append(tracker)
        virtual = None
        if args.virtual_cam is True:
            virtual = VirtualCamera(args.width, args.height)
        elif args.virtual_cam is None:
            virtual = discover_virtual_camera(args.width, args.height)
        if virtual: resources.append(virtual)
        preview = Preview(mirror=args.mirror_preview) if args.preview else None
        if preview: resources.append(preview)
        capture = Capture(args.camera, args.width, args.height)
        resources.append(capture)
        while True:
            sample = capture.read()
            if sample is None: break
            frame, timestamp = sample
            if args.mirror_capture:
                import cv2
                frame = cv2.flip(frame, 1)
            tracker.calibrating = calibration.active
            tracked = tracker.track(frame, timestamp)
            candidate = calibration.update(tracked.observation if tracked else None)
            if candidate:
                geometry = candidate
                tracker.geometry = pipeline.geometry = geometry
                calibration.geometry = geometry
            output = pipeline.process(frame, tracked)
            if virtual: virtual.send(output)
            count += 1
            if preview:
                text = f"{provider} | {count/max(.001,clock()-started):.1f} fps | g correction c nudge Space Calibration q quit"
                if overlay: text = f"Camera {geometry.camera_pos}, focal {geometry.focal_length:.0f} | arrows XY +/- Z [] focal r reset"
                if not overlay and calibration.state in ("failed", "timeout", "cancelled", "success"): text = calibration.message
                key = preview.show(output, text, calibration)
                if key == ord('q'): break
                if key == 27: calibration.cancel()
                elif key == ord('g'): pipeline.enabled = not pipeline.enabled
                elif key == ord('c'): overlay = not overlay
                elif key == ord(' '): calibration.start()
                elif overlay and key >= 0:
                    updated = nudge(geometry, key)
                    if updated != geometry:
                        save_config(args.config, updated)
                        geometry = updated
                        tracker.geometry = pipeline.geometry = geometry
                        calibration.geometry = geometry
    except KeyboardInterrupt:
        pass
    finally:
        errors = []
        for resource in reversed(resources):
            try: resource.close()
            except Exception as error: errors.append(error)
        if errors: raise errors[0]


if __name__ == "__main__":
    main()
