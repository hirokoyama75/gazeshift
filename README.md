# gazeshift v2.0

gazeshift adjusts your eyes in webcam video so looking at your screen feels closer to looking toward the camera. It preserves natural eye movement and can send the result to video calls through OBS Virtual Camera.

This rebuilds [WangWilly's gaze-correction-cam](https://github.com/WangWilly/gaze-correction-cam) using [Chih-Fan Hsu's original left and right eye models](https://github.com/chihfanhsu/gaze_correction/tree/7cc76e4ccaf76c950730c593ffa22bed30c807a4), without retraining them.

## What changed

- Both eyes run together through ONNX Runtime, replacing two TensorFlow sessions.
- Smoother tracking reduces jitter. Correction fades during blinks and large head turns.
- Softer edges and smoother fades avoid rectangular eye patches and doubled irises.
- Two-look calibration adjusts for your camera's position and saves the result.
- Capture keeps the newest frame; preview controls stay off the call feed.
- Desktop controls offer saved dark/light themes, strength and aim adjustments.
- Eye compositing converts only the region it samples, reducing CPU memory traffic.

## Run it

Targets Windows 11 x64 and macOS 14+ on Apple Silicon. Use Python 3.12 through **uv**; Intel Macs and Windows ARM64 are unsupported.

On Windows, double-click **Start Gazeshift.bat** (uv must be installed). Or launch the desktop interface with:

```powershell
uv run --exact --extra virtualcam gui.py
```

Choose your camera, then press **Start camera**. Use the **Correction** tab for strength, aim adjustments and calibration. **Show original** compares the untouched picture without changing the call feed. Your controls are saved automatically; the camera stays off until you start it.

Use the **Dark / Light** selector at the top to change appearance. Dark is the default.

Strength runs from 0–150%: up to 100% fades the correction; above 100% requests a larger shift within the same ±15° limits. This does not lock your gaze. Blinks and head-turn fades remain active.

Horizontal aim adds a left/right adjustment to both eyes; vertical aim adds an up/down adjustment. Start at zero, calibrate, then make small changes if needed. These are offsets, not a fixed gaze target. Mirroring reverses the apparent horizontal direction in the preview.

The eye model is included; the face-tracking model downloads on first use. Camera calibration lives in `~/.gaze_cam/config.json`, and desktop preferences in `~/.gaze_cam/desktop.json`.

The regular launcher uses CPU. For AMD or Intel graphics on Windows, double-click **Start Gazeshift AMD GPU.bat**. It installs DirectML separately and selects **AMD / Intel GPU**, preserving the CPU environment. Or use:

```powershell
uv run --exact --no-group cpu --extra dml --extra virtualcam gui.py --provider dml
```

For NVIDIA CUDA, replace `--extra dml` with `--extra cuda` and `--provider dml` with `--provider cuda`; compatible CUDA/cuDNN libraries are required. Available options appear under **Call output & setup**. GPU acceleration moves eye inference to the GPU; tracking, compositing and preview still use CPU.

For video calls, install OBS and enable **Send video to call apps** before starting the camera. Choose **OBS Virtual Camera** in your call app; leave OBS's own output stopped. Close OBS if it holds your webcam. On macOS, first start and stop OBS's Virtual Camera once to approve its camera extension.

## Command-line option

The original CLI remains available: `uv run --exact app.py --provider cpu`. Add `--extra virtualcam` before `app.py` and `--virtual-cam` after it for call output. Its preview shortcuts are:

- **Space:** calibrate by looking at the lens, then the on-screen crosshair. **Escape** cancels.
- **g:** toggle correction. **q:** quit.
- **c:** show manual adjustments. Arrows move camera X/Y, `+/-` adjusts depth, `[/]` adjusts focal length, and **r** resets defaults.

Controls need the preview window; otherwise, Ctrl+C quits. Failed or cancelled calibration preserves saved settings.

## Verification

```powershell
uv run --exact --group dev python scripts/verify_headless.py --all --timeout-seconds 900 --report artifacts/verification.json
```

This rebuilds the model from the included original checkpoints and compares it against TensorFlow in a separate environment. It also tests the video pipeline and calibration without opening a camera or window. Datasets, recordings, reports and local environments stay out of Git.

On October 14, 2026, **49/49 headless tests passed**, with no skips or expected failures. Both-eye CPU inference measured **7.11 ms p95** on a Ryzen 5 3600, below the **9 ms** gate (50 warmups, 500 timed runs). A separate real-window test covers desktop controls, theme switching and calibration placement without opening a camera. Production does not pin CPU affinity.

Live Windows/Mac camera and OBS checks, sustained 30 fps, capture-to-output latency, visual quality, real-person calibration and memory usage remain **manual_pending**.

A separate RX 6600 XT benchmark measured **0.95 ms p95** dual-eye DirectML inference versus **7.00 ms** CPU (ORT 1.24.4, 50 warmups / 500 runs). All eight real-data/angle-sweep comparisons passed the unchanged TensorFlow accuracy limit; profiling confirmed GPU execution. This does not establish live FPS or power consumption. Synthetic 720p two-eye compositing improved from **13.22 to 5.02 ms median**, with identical pixels in that benchmark. No model weights or correction limits changed.

## Credits and license

Original code and checkpoints: **Chih-Fan Hsu, 2019**, under [LICENSE](LICENSE). Source and asset hashes are in `scripts/source_manifest.json`. Older modules remain for reference; use `gui.py` or `app.py`.

Downloaded DIRL data stays local with the [publisher's notice](https://sites.google.com/site/chihfanhsuwebsite/dataset). Optional virtual-camera support uses GPL-2.0 `pyvirtualcam` and is excluded from the default install.
