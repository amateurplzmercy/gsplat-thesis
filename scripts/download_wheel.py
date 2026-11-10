"""Resumable HTTPS wheel download with byte-range validation and optional SHA256."""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def download(url, target, expected_sha=None, workers=6):
    clean_url, fragment = urllib.parse.urldefrag(url)
    if not clean_url.startswith("https://"):
        raise ValueError("HTTPS is required")
    expected_sha = expected_sha or urllib.parse.parse_qs(fragment).get("sha256", [None])[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with target.open("rb") as stream:
            hasher = hashlib.sha256()
            for data in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(data)
            digest = hasher.hexdigest()
        if expected_sha and digest != expected_sha:
            raise ValueError("Existing file does not match expected SHA256")
        print(f"Already downloaded: {target.name} sha256={digest}", flush=True)
        return
    request = urllib.request.Request(clean_url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith("bytes 0-0/"):
            raise RuntimeError("Server did not honor exact byte-range request")
        total = int(content_range.rsplit("/", 1)[1])
        identity = response.headers.get("ETag")
        response.read()
    parts = target.with_suffix(target.suffix + ".parts")
    parts.mkdir(exist_ok=True)
    meta = {"url": clean_url, "bytes": total, "expected_sha256": expected_sha, "etag": identity}
    metadata = parts / "source.json"
    if metadata.exists() and json.loads(metadata.read_text()) != meta:
        raise RuntimeError("Saved download refers to a different source; inspect parts first")
    metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    chunk_size = 8 * 1024 * 1024
    ranges = [(start, min(total - 1, start + chunk_size - 1))
              for start in range(0, total, chunk_size)]

    def fetch(bounds):
        start, end = bounds
        size = end - start + 1
        path = parts / f"{start:012d}.part"
        if path.exists() and path.stat().st_size == size:
            return size
        for attempt in range(4):
            try:
                headers = {"Range": f"bytes={start}-{end}"}
                if identity:
                    headers["If-Range"] = identity
                req = urllib.request.Request(clean_url, headers=headers)
                with urllib.request.urlopen(req, timeout=40) as source:
                    if source.status != 206 or source.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise RuntimeError(f"Unexpected byte range for {start}")
                    with path.open("wb") as stream:
                        received = 0
                        while data := source.read(256 * 1024):
                            received += len(data)
                            if received > size:
                                raise RuntimeError("Response exceeded requested range")
                            stream.write(data)
                if path.stat().st_size != size:
                    raise RuntimeError("Incomplete range response")
                return size
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    done = 0
    started = time.monotonic()
    print(f"Downloading {target.name}: {total / 2**20:.1f} MiB in {len(ranges)} resumable ranges", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, bounds) for bounds in ranges]):
            done += future.result()
            if done == total or done // (128 * 2**20) != (done - chunk_size) // (128 * 2**20):
                print(f"{done / 2**20:.1f}/{total / 2**20:.1f} MiB, {time.monotonic()-started:.1f}s", flush=True)
    staged = target.with_suffix(target.suffix + ".assembling")
    digest = hashlib.sha256()
    with staged.open("wb") as output:
        for start, _ in ranges:
            with (parts / f"{start:012d}.part").open("rb") as source:
                while data := source.read(1024 * 1024):
                    digest.update(data)
                    output.write(data)
    actual = digest.hexdigest()
    if expected_sha and actual != expected_sha:
        raise RuntimeError(f"SHA256 mismatch: {actual}")
    staged.replace(target)
    meta.update({"sha256": actual, "hash_verified_against_source": bool(expected_sha)})
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Completed {target.name}: sha256={actual}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("target", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    download(args.url, args.target, args.sha256, args.workers)
