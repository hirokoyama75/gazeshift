# gazeshift v2.0

gazeshift adjusts your eyes in webcam video so looking at your screen feels closer to looking toward the camera. It preserves natural eye movement and can send the result to video calls through OBS Virtual Camera.

This rebuilds [WangWilly's gaze-correction-cam](https://github.com/WangWilly/gaze-correction-cam) using [Chih-Fan Hsu's original left and right eye models](https://github.com/chihfanhsu/gaze_correction/tree/7cc76e4ccaf76c950730c593ffa22bed30c807a4), without retraining them.

## What changed

- Both eyes run together through ONNX Runtime, replacing two TensorFlow sessions.
- Smoother tracking reduces jitter. Correction fades during blinks and large head turns.
- Softer edges and smoother fades avoid rectangular eye patches and doubled irises.
- Two-look calibration adjusts for your camera's position and saves the result.
- Capture keeps the newest frame; preview controls stay off the call feed.

## Run it

Targets Windows 11 x64 and macOS 14+ on Apple Silicon. Use Python 3.12 through **uv**; Intel Macs and Windows ARM64 are unsupported.

```powershell
uv sync
uv run --exact app.py --provider cpu
```

The eye model is included; the face-tracking model downloads on first launch. Options include `--camera 1`, `--mirror-preview`, and `--no-preview`. Settings live in `~/.gaze_cam/config.json`; `--config` chooses another file.

For an AMD or Intel GPU on Windows, try DirectML:

```powershell
uv run --exact --no-group cpu --extra dml app.py --provider dml
```

For NVIDIA CUDA, replace `dml` with `cuda` in both places. Use uv's matching selectors so only one ONNX Runtime package is installed.

For video calls, install OBS and run:

```powershell
uv run --exact --extra virtualcam app.py --virtual-cam
```

Choose **OBS Virtual Camera** in your call app. On macOS, first start and stop OBS's Virtual Camera once to approve its camera extension.

## Controls

- **Space:** calibrate by looking at the lens, then the on-screen crosshair. **Escape** cancels.
- **g:** toggle correction. **q:** quit.
- **c:** show manual adjustments. Arrows move camera X/Y, `+/-` adjusts depth, `[/]` adjusts focal length, and **r** resets defaults.

Controls need the preview window; otherwise, Ctrl+C quits. Failed or cancelled calibration preserves saved settings.

## Verification

```powershell
uv run --exact --group dev python scripts/verify_headless.py --all --timeout-seconds 900 --report artifacts/verification.json
```

This rebuilds the model from the included original checkpoints and compares it against TensorFlow in a separate environment. It also tests the video pipeline and calibration without opening a camera or window. Datasets, recordings, reports and local environments stay out of Git.

On October 14, 2026, **44/44 tests passed**, with no skips. Both-eye CPU inference measured **7.22 ms p95** on a Ryzen 5 3600, below the **9 ms** gate (50 warmups, 500 timed runs). Production does not pin CPU affinity.

Live Windows/Mac camera and OBS checks, sustained 30 fps, capture-to-output latency, visual quality, real-person calibration and memory usage remain **manual_pending**.

## Credits and license

Original code and checkpoints: **Chih-Fan Hsu, 2019**, under [LICENSE](LICENSE). Source and asset hashes are in `scripts/source_manifest.json`. Older modules remain for reference; run `app.py`.

Downloaded DIRL data stays local with the [publisher's notice](https://sites.google.com/site/chihfanhsuwebsite/dataset). Optional virtual-camera support uses GPL-2.0 `pyvirtualcam` and is excluded from the default install.
