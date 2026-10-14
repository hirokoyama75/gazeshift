"""One-process, arm64 app; no per-launch extraction or browser runtime."""
from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).resolve().parent
generated = root / ".cache/macos-build/assets"
datas = collect_data_files("mediapipe") + [
    (str(root / "weights/flx_12.onnx"), "weights"),
    (str(root / "weights/face_landmarker.task"), "weights"),
    (str(root / "LICENSE"), "."),
    (str(generated / "build-info.json"), "."),
    (str(generated / "licenses"), "licenses"),
]
for package in ("mediapipe", "onnxruntime", "pyvirtualcam"):
    datas += copy_metadata(package)
binaries = collect_dynamic_libs("mediapipe")
a = Analysis([str(root / "scripts/desktop_entry.py")], pathex=[str(root)],
             binaries=binaries, datas=datas, hiddenimports=["pyvirtualcam"],
             excludes=["tensorflow", "torch", "jax", "IPython", "pytest", "onnx"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Gazeshift",
          debug=False, strip=False, upx=False, console=False, argv_emulation=False,
          target_arch="arm64", codesign_identity=os.environ.get("GAZESHIFT_CODESIGN_IDENTITY"),
          entitlements_file=str(root / "packaging/macos-entitlements.plist"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Gazeshift")
app = BUNDLE(coll, name="Gazeshift.app", bundle_identifier="com.hirokoyama.gazeshift",
             version="2.0.0", info_plist={
                 "CFBundleDisplayName": "Gazeshift",
                 "CFBundleShortVersionString": "2.0.0",
                 "LSMinimumSystemVersion": "14.0",
                 "NSPrincipalClass": "NSApplication",
                 "NSHighResolutionCapable": True,
                 "NSCameraUsageDescription": "Gazeshift adjusts eye contact in your camera video. Your video is processed on this Mac.",
             })
