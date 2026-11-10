"""Prepare a project-local Python runtime; no installer, registry or global PATH changes."""

import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "downloads"
RUNTIME = ROOT / "tools" / "python310"


def fetch(url, filename, expected_sha=None, offline=False):
    target = DOWNLOADS / filename
    partial = target.with_suffix(target.suffix + ".part")
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    source_path = target
    if not target.exists():
        if offline:
            raise FileNotFoundError(f"Missing cached bootstrap file: {target}")
        print(f"Downloading {filename}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as dest:
            while chunk := source.read(1024 * 1024):
                dest.write(chunk)
        source_path = partial
    with source_path.open("rb") as stream:
        hasher = hashlib.sha256()
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
        digest = hasher.hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"SHA256 mismatch for {source_path}")
    if source_path == partial:
        partial.replace(target)
    entry = {"url": url, "filename": filename, "size_bytes": target.stat().st_size,
             "sha256": digest, "checked_at_unix": time.time(),
             "hash_provenance": "locally_computed_for_reproduction"}
    with (DOWNLOADS / "bootstrap_manifest.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runtime-only", action="store_true", help="Prepare the ZIP runtime without fetching get-pip.py")
    parser.add_argument("--sha256", help="Expected SHA256 of the Python archive")
    args = parser.parse_args()
    package = fetch("https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip",
                    "python-3.10.11-embed-amd64.zip", expected_sha=args.sha256, offline=args.offline)
    if not args.runtime_only:
        fetch("https://bootstrap.pypa.io/get-pip.py", "get-pip.py", offline=args.offline)
    if not (RUNTIME / "python.exe").exists():
        RUNTIME.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(package) as archive:
            for name in archive.namelist():
                if not (RUNTIME / name).resolve().is_relative_to(RUNTIME.resolve()):
                    raise ValueError(f"Unexpected archive path: {name}")
            archive.extractall(RUNTIME)
    # Embeddable Python is intentionally isolated. All dependencies are vendored
    # in this runtime, rather than treated as a general-purpose Python installation.
    (RUNTIME / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (RUNTIME / "python310._pth").write_text(
        f"python310.zip\n.\nLib/site-packages\n{ROOT}\nimport site\n", encoding="utf-8")
    print(f"Prepared runtime: {RUNTIME / 'python.exe'}", flush=True)


if __name__ == "__main__":
    main()
