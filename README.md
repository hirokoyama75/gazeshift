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
- Automatic processing compares warmed-up backend timings; the preview shows the backend actually used.
- Mac cameras use native device names, and the interface uses the system's font.
- The standalone Mac build includes Python and both models; idle polling slows down and minimized previews stop rendering.
- Eye compositing converts only the region it samples, reducing CPU memory traffic.

## Run it

Targets Windows 11 x64 and macOS 14+ on Apple Silicon. Use Python 3.12 through **uv**; Intel Macs and Windows ARM64 are unsupported.

On Windows, double-click **Start Gazeshift.bat** (uv must be installed). Or launch the desktop interface with:

```powershell
uv run --exact --extra virtualcam gui.py
```

For a standalone Mac build, unzip **Gazeshift-macOS-arm64.zip**, move **Gazeshift.app** into Applications, and open it. It includes Python, both models and call-output support; launching it needs neither uv nor a network connection. Allow **Gazeshift** under **System Settings → Privacy & Security → Camera**, then press **Start camera**. OBS is still required for video-call output.

[Download the October 14 development build](https://github.com/hirokoyama75/gazeshift/actions/runs/38605237108/artifacts/11914929233) (GitHub sign-in required). Expand the artifact download, then its `Gazeshift-macOS-arm64.zip` to get the app.

Development builds are ad-hoc signed, with camera permission metadata and entitlements. Developer ID signing and Apple notarization are still required before a normal public download release. The local build accepts `GAZESHIFT_CODESIGN_IDENTITY` for an installed signing certificate.

If macOS blocks this development build, use **System Settings → Privacy & Security → Open Anyway** for Gazeshift; see [Apple's app-opening instructions](https://support.apple.com/en-ie/102445).

To run directly from source on Apple Silicon macOS 14+, install uv, open Terminal in this project folder, and run:

```bash
chmod +x "Start Gazeshift.command"
./"Start Gazeshift.command"
```

After that, double-click **Start Gazeshift.command**. It keeps its environment in `.cache/mac`. Allow Terminal or Python in **System Settings → Privacy & Security → Camera** when prompted; restart Gazeshift after changing permissions. Intel Macs and running Terminal under Rosetta are unsupported.

New desktop settings start with **Automatic** processing. It compares CPU and installed accelerators using 10 warmups followed by 20 timed calls per backend, choosing the lowest p95. Saved choices are respected. Core ML uses static shapes, full-precision GPU accumulation and a model-content cache; the CPU may still be faster for this small model. You can select **CPU** or **Apple Core ML** explicitly under **Setup**. First-time Core ML compilation can delay startup. These choices use the existing ONNX model without changing its weights.

Choose your camera, then press **Start camera**. Use the **Correction** tab for strength, aim adjustments and calibration. **Show original** compares the untouched picture without changing the call feed. Your controls are saved automatically; the camera stays off until you start it.

Use the **Dark / Light** selector at the top to change appearance. Dark is the default; both use a neutral palette and the native system font.

Strength runs from **0–250%**, including 200% and 250%. Up to 100% fades the correction; above 100% multiplies the requested shift within the same **±15° model limits**. Once an angle reaches that limit, increasing the percentage cannot move it further. This uses the same single both-eye inference call per processed frame; it does not add model passes. Blinks and head-turn fades remain active.

Looking at an upper monitor changes the direction of your eyes. The current engine applies a calibrated screen/camera offset, without continuously measuring your iris direction or identifying which display you are looking at. More strength or a vertical aim adjustment can help within its range, but 250% does not guarantee invisible switching between screens or a constant camera-facing gaze.

Horizontal aim adds a left/right adjustment to both eyes; vertical aim adds an up/down adjustment. Start at zero, calibrate, then make small changes if needed. These are offsets, not a fixed gaze target. Mirroring reverses the apparent horizontal direction in the preview.

The eye model is included; the face-tracking model downloads on first use. Camera calibration lives in `~/.gaze_cam/config.json`, and desktop preferences in `~/.gaze_cam/desktop.json`.

The regular launcher uses CPU. For AMD or Intel graphics on Windows, double-click **Start Gazeshift AMD GPU.bat**. It installs DirectML separately and selects **AMD / Intel GPU**, preserving the CPU environment. Or use:

```powershell
uv run --exact --no-group cpu --extra dml --extra virtualcam gui.py --provider dml
```

For NVIDIA on Windows, double-click **Start Gazeshift NVIDIA GPU.bat**. It installs ONNX Runtime with its CUDA/cuDNN libraries in `.cache/cuda`, separate from CPU and AMD. The first download is large. A compatible NVIDIA driver is still required. CUDA is preloaded before session creation, TF32 is disabled to preserve fp32 accuracy, and a failed CUDA startup produces an error instead of reporting CPU work as NVIDIA acceleration. Or use:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".cache/cuda"
uv run --locked --exact --no-group cpu --extra cuda --extra virtualcam gui.py --provider cuda
```

Available options appear under **Setup**. GPU acceleration moves eye inference to the GPU; tracking, compositing and preview still use CPU. Mac and NVIDIA hardware performance has not yet been measured.

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

On October 14, 2026, **62/62 headless tests passed**, with no skips or expected failures, including platform/backend and packaging checks. The separate real-window test also passed, covering desktop controls, theme switching, calibration placement, automatic backend labels, preview image reuse and minimized rendering without opening a camera. The CPU gate remains **9 ms p95** (50 warmups, 500 timed runs). Production does not pin CPU affinity.

Live Windows/Mac camera and OBS checks, sustained 30 fps, capture-to-output latency, visual quality, real-person calibration and memory usage remain **manual_pending**. The Mac and NVIDIA additions have simulated platform regression checks; those do not establish hardware performance or live acceptance.

To measure a target machine without opening its camera, run the matched backend benchmark:

```bash
# Apple Silicon: default ONNX Runtime includes the Core ML provider.
UV_PROJECT_ENVIRONMENT=.cache/mac uv run --locked --exact python scripts/benchmark_provider.py --provider coreml --report artifacts/coreml.json
```

```powershell
# NVIDIA: use the same isolated runtime as its launcher.
$env:UV_PROJECT_ENVIRONMENT = ".cache/cuda"
uv run --locked --exact --no-group cpu --extra cuda python scripts/benchmark_provider.py --provider cuda --report artifacts/cuda.json
```

Each report includes matched CPU/backend timings (50 warmups, 500 samples), the model hash, accuracy checks, and a separate execution profile confirming backend work. With the local verification fixtures present, it checks all eight real-data/angle cases against the original TensorFlow outputs at the unchanged `0.0001` limit. A fresh clone instead checks three synthetic eye cases against CPU and explicitly labels original-model parity unverified; run the full verifier first for that stronger check. Reports and profiles stay local. Benchmark timings measure eye inference, not total webcam latency or live FPS.

## Build the standalone Mac app

Build on an Apple Silicon Mac with macOS 14+ and native Python 3.12:

```bash
UV_PROJECT_ENVIRONMENT=.cache/mac-build-env uv run --locked --exact --group build --extra virtualcam python scripts/build_macos.py
```

The builder checks the pinned models, bundles the existing Tk interface and processing libraries with PyInstaller, verifies arm64 code signing, runs the packaged executable from outside the project with camera I/O forbidden, and creates `artifacts/macos/Gazeshift-macOS-arm64.zip`. It also saves a source archive, bundled dependency notices and measured CPU/Core ML reports. The smoke check loads the real tracker, creates the real interface with the camera off, and checks backend accuracy and profiling. Camera/OBS operation and sustained resource usage still require live Mac testing.

The **Build macOS app** GitHub Actions workflow uses an Apple Silicon runner and uploads these development artifacts. It runs on the `codex/gazeshift-v2` branch or by manual dispatch. End users do not install the build tools.

The `.app` uses one process, without a browser runtime or per-launch unpacking. Its processing pipeline, model precision, latest-frame queues and correction limits are unchanged. The interface polls at 4 Hz while stopped and skips preview rendering while minimized; call output continues normally. On macOS, ONNX workers sleep between inference calls while retaining fast scheduling within each run. Compiled Core ML caches live in `~/Library/Caches/Gazeshift`, outside the signed app. Preferences remain in `~/.gaze_cam/`. The native Quit command uses the same camera cleanup as closing the window.

On the shared M1 macOS 14 runner, the normal packaged interface with its camera off measured **0.20% of one CPU core and 109 MiB RSS** over five seconds after a ten-second startup warmup. This measures idle use only. In a separate 90-frame synthetic inference test paced at 30 FPS, sleeping workers between calls reduced CPU time by about **14%** versus the default wait policy (151% → 130% of one core), with delivery rates of 29.3 versus 29.9 FPS. Shared-runner timings vary; this does not establish whole-pipeline camera FPS, battery use, or electrical power. Both CPU and Core ML passed the packaged synthetic consistency checks; original TensorFlow parity was checked separately by the Windows headless verifier.

A separate RX 6600 XT benchmark measured **0.95 ms p95** dual-eye DirectML inference versus **7.00 ms** CPU (ORT 1.24.4, 50 warmups / 500 runs). All eight real-data/angle-sweep comparisons passed the unchanged TensorFlow accuracy limit; profiling confirmed GPU execution. This does not establish live FPS or power consumption. Synthetic 720p two-eye compositing improved from **13.22 to 5.02 ms median**, with identical pixels in that benchmark. No model weights or correction limits changed.

## Credits and license

Original code and checkpoints: **Chih-Fan Hsu, 2019**, under [LICENSE](LICENSE). Source and asset hashes are in `scripts/source_manifest.json`. Older modules remain for reference; use `gui.py` or `app.py`.

Downloaded DIRL data stays local with the [publisher's notice](https://sites.google.com/site/chihfanhsuwebsite/dataset). Optional virtual-camera support uses GPL-2.0 `pyvirtualcam` and is excluded from the default install.
