"""Deterministic DIRL inputs; decode only after validating the pinned Git blob."""
import json
import pickle
import numpy as np
from scripts.assets import ROOT, git_blob, sha256


def materialize():
    manifest = json.loads((ROOT / "scripts/source_manifest.json").read_text())
    data = {}
    for side in ("l", "r"):
        path = ROOT / f".cache/source/0008_{side.upper()}"
        entry = next(e for e in manifest["assets"] if e["local"] == path.relative_to(ROOT).as_posix())
        raw = path.read_bytes()
        if git_blob(raw) != entry["git_blob"]:
            raise ValueError("Unverified DIRL pickle")
        decoded = pickle.loads(raw, encoding="latin1")
        required = {"img", "anchor_map", "p", "v", "h", "ap", "origin_xy", "ratio"}
        if set(decoded) != required or len({len(v) for v in decoded.values()}) != 1:
            raise ValueError("Incompatible DIRL schema")
        data[side] = decoded
    cases = []
    labels = [(f"dirl_{s}_to_{t}", s, t, None) for s, t in manifest["pairs"]]
    labels += [(f"angle_sweep_{v:+d}_{h:+d}", 0, 0, (v, h)) for v, h in manifest["sweeps"]]
    for label, source, target, override in labels:
        inputs = {}
        for side, decoded in data.items():
            image = np.asarray(decoded["img"][source])
            anchors = np.asarray(decoded["anchor_map"][source])
            if image.shape != (48, 64, 3) or image.dtype != np.uint8 or anchors.shape != (48, 64, 14):
                raise ValueError("Unexpected image/anchor schema")
            angle = override or (decoded["v"][target] - decoded["v"][source], decoded["h"][target] - decoded["h"][source])
            inputs[f"img_{side}"] = image[None].astype(np.float32) / 255
            inputs[f"fp_{side}"] = anchors[None, ..., :12].astype(np.float32)
            inputs[f"ang_{side}"] = np.asarray([angle], dtype=np.float32)
        cases.append({"label": label, "source": source, "target": target, "inputs": inputs})
    fixture = {"schema": 1, "seed": 0, "manifest_sha256": sha256(ROOT / "scripts/source_manifest.json"), "cases": cases}
    path = ROOT / ".cache/parity/dirl_flx12.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(fixture, protocol=4))
    return fixture
