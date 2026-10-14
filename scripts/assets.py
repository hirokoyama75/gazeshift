"""Acquire pinned, blob-verified assets before decoding any upstream pickle."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "7cc76e4ccaf76c950730c593ffa22bed30c807a4"
NOTICE = "https://sites.google.com/site/chihfanhsuwebsite/dataset"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch(url, path, blob=None, deadline=None):
    path = Path(path)
    if path.exists() and (blob is None or git_blob(path.read_bytes()) == blob):
        return path
    for attempt in range(3):
        remaining = (deadline - time.monotonic()) if deadline else 60
        if remaining <= 0:
            raise TimeoutError(url)
        try:
            with urllib.request.urlopen(url, timeout=min(30, remaining)) as response:
                data = response.read()
            if blob and git_blob(data) != blob:
                raise ValueError(f"Git blob mismatch: {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            return path
        except Exception:
            if attempt == 2:
                raise


def acquire(deadline=None):
    manifest = json.loads((ROOT / "scripts/source_manifest.json").read_text())
    hashes = {}
    for entry in manifest["assets"]:
        path = ROOT / entry["local"]
        fetch(f"https://raw.githubusercontent.com/chihfanhsu/gaze_correction/{COMMIT}/{entry['path']}",
              path, entry["git_blob"], deadline)
        hashes[entry["local"]] = sha256(path)
    notice = ROOT / ".cache/parity/DATASET_NOTICE.txt"
    notice.parent.mkdir(parents=True, exist_ok=True)
    notice.write_text("DIRL published by Chih-Fan Hsu. Dataset is available for non-commercial use; cite the paper when used.\n" +
        "Hsu, Chih-Fan; Wang, Yu-Shuen; Lei, Chin-Laung; Chen, Kuan-Ta (2019). Look at Me: Correcting Eye Gaze in Live Video Communication.\n" +
        "ACM Transactions on Multimedia Computing, Communications, and Applications 15(2), Article 38. DOI: 10.1145/3311784.\n" +
        "Publisher's notice and dataset terms: " + NOTICE + "\nCached for local validation; do not redistribute the dataset.\n")
    page = fetch(NOTICE, ROOT / ".cache/parity/DATASET_NOTICE.html", deadline=deadline)
    if b"non-commercial" not in page.read_bytes():
        raise ValueError("Publisher's dataset notice is missing or incompatible")
    hashes[page.relative_to(ROOT).as_posix()] = sha256(page)
    (ROOT / ".cache/source/hashes.json").write_text(json.dumps(hashes, indent=2))
    return hashes


if __name__ == "__main__":
    print(json.dumps(acquire(time.monotonic() + 240), indent=2))
