"""List and extract selected members of the authors' pretrained model archive."""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath


URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip"
MIB = 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


class RangeArchive:
    """Read only requested ZIP ranges, with a bounded transfer size."""

    def __init__(self, url):
        self.url = url
        self.total = None
        self.etag = None
        self.position = 0
        self.transferred = 0
        self._fetch(0, 0)

    def _fetch(self, start, end):
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Range"] = self.etag
        for attempt in range(3):
            try:
                request = urllib.request.Request(self.url, headers=headers)
                with urllib.request.urlopen(request, timeout=25) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         response.headers.get("Content-Range", ""))
                    if response.status != 206 or not match:
                        raise ValueError("Server did not return a partial ZIP response")
                    first, last, total = map(int, match.groups())
                    if (first, last) != (start, end) or self.total not in (None, total):
                        raise ValueError("Archive range or total size changed")
                    data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise OSError("Incomplete ZIP range")
                    self.total = total
                    self.etag = response.headers.get("ETag")
                    self.transferred += len(data)
                    return data
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    def seek(self, offset, whence=os.SEEK_SET):
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.total}[whence]
        self.position = base + offset
        if self.position < 0:
            raise OSError("Negative ZIP offset")
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        remaining = max(0, self.total - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if size > 16 * MIB:
            raise ValueError("Refusing an unbounded ZIP read")
        if not size:
            return b""
        start = self.position
        data = self._fetch(start, start + size - 1)
        self.position += size
        return data


def selected_members(archive):
    scenes = {}
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.name == "cameras.json" and len(path.parts) >= 2:
            scenes.setdefault(path.parts[-2], {})["cameras.json"] = member
        if path.parts[-3:] == ("point_cloud", "iteration_30000", "point_cloud.ply") and len(path.parts) >= 4:
            scenes.setdefault(path.parts[-4], {})["point_cloud.ply"] = member
    return {name: files for name, files in sorted(scenes.items()) if len(files) == 2}


def info(member):
    return {"archive_member": member.filename, "bytes": member.file_size,
            "compressed_bytes": member.compress_size, "zip_crc32": f"{member.CRC:08x}"}


def extract_member(archive, member, target):
    staged = target.with_suffix(target.suffix + ".partial")
    sha256 = hashlib.sha256()
    crc = count = 0
    if target.exists():
        source = target.open("rb")
        destination = None
    else:
        source = archive.open(member)
        destination = staged.open("wb")
    try:
        with source:
            while data := source.read(4 * MIB):
                if destination:
                    destination.write(data)
                sha256.update(data)
                crc = zlib.crc32(data, crc)
                count += len(data)
    finally:
        if destination:
            destination.close()
    if count != member.file_size or crc != member.CRC:
        raise ValueError(f"Size or CRC mismatch for {target.name}")
    if destination:
        staged.replace(target)
    return {**info(member), "sha256": sha256.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene")
    parser.add_argument("--max-mib", type=int, default=512)
    args = parser.parse_args()
    remote = RangeArchive(URL)
    with zipfile.ZipFile(remote) as archive:
        scenes = selected_members(archive)
        if args.scene is None:
            report = {"url": URL, "archive_bytes": remote.total, "etag": remote.etag,
                      "scenes": {key: {name: info(member) for name, member in files.items()}
                                 for key, files in scenes.items()}}
            (ROOT / "reports" / "graphdeco_archive_index.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            for key, files in scenes.items():
                model = files["point_cloud.ply"]
                print(f"{key:15} {model.file_size / MIB:7.1f} MiB PLY, "
                      f"{model.compress_size / MIB:7.1f} MiB transfer, paired cameras")
            return
        members = scenes.get(args.scene)
        if not members or not re.fullmatch(r"[A-Za-z0-9_-]+", args.scene):
            parser.error("Choose an available scene from the archive index")
        if sum(member.file_size for member in members.values()) > args.max_mib * MIB:
            parser.error("Selected scene exceeds the size limit")
        folder = ROOT / "data" / args.scene
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, member in members.items():
            print(f"Reading {member.filename} ({member.compress_size / MIB:.1f} MiB)", flush=True)
            files[name] = extract_member(archive, member, folder / name)
        manifest = {"source_url": URL, "archive_etag": remote.etag, "archive_bytes": remote.total,
                    "scene": args.scene, "iteration": 30000, "network_bytes": remote.transferred,
                    "files": files}
        (folder / "source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
