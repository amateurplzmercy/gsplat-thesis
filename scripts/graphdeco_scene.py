"""Read standard Graphdeco Gaussian PLYs and their matching camera export."""

import json
import math
import re
from pathlib import Path

import numpy as np


def load_gaussians(path: Path):
    fields = []
    count = 0
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Expected a PLY file")
        if stream.readline(1024).strip() != b"format binary_little_endian 1.0":
            raise ValueError("Expected binary little-endian PLY 1.0")
        for _ in range(256):
            line = stream.readline(1024).decode("ascii").strip().split()
            if not line:
                raise ValueError("Incomplete or invalid PLY header")
            if line[0] in ("comment", "obj_info"):
                continue
            if line[:2] == ["element", "vertex"] and len(line) == 3 and not count:
                count = int(line[2])
            elif line[:2] in (["property", "float"], ["property", "float32"]) and len(line) == 3 and count > 0:
                fields.append(line[2])
            elif line == ["end_header"]:
                break
            else:
                raise ValueError(f"Unsupported Graphdeco PLY header line: {line}")
        else:
            raise ValueError("PLY header exceeded limit")
        offset = stream.tell()
    if count < 1 or len(fields) != len(set(fields)):
        raise ValueError("Invalid vertex count or duplicate PLY fields")
    dtype = np.dtype([(name, "<f4") for name in fields])
    if offset + dtype.itemsize * count != path.stat().st_size:
        raise ValueError("PLY vertex payload size does not match header")

    rest_count = sum(name.startswith("f_rest_") for name in fields)
    side = math.isqrt(1 + rest_count // 3)
    if rest_count != 3 * (side * side - 1) or side > 5:
        raise ValueError("Invalid spherical harmonic property count")
    bases = side * side
    records = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(count,))

    def columns(names):
        return np.stack([records[name] for name in names], axis=-1)

    colors = np.empty((count, bases, 3), dtype=np.float32)
    for channel in range(3):
        colors[:, 0, channel] = records[f"f_dc_{channel}"]
        for basis in range(1, bases):
            colors[:, basis, channel] = records[f"f_rest_{channel * (bases - 1) + basis - 1}"]
    logits = records["opacity"]
    arrays = {
        "means": columns(("x", "y", "z")),
        "quats": columns(f"rot_{i}" for i in range(4)),
        "scales": np.exp(columns(f"scale_{i}" for i in range(3))),
        "opacities": np.exp(-np.logaddexp(0, -logits)),
        "colors": colors,
    }
    lengths = np.linalg.norm(arrays["quats"], axis=1)
    if np.any(lengths <= 0) or np.any(arrays["scales"] <= 0):
        raise ValueError("Degenerate quaternion or Gaussian scale")
    arrays["quats"] /= lengths[:, None]
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Non-finite Gaussian parameters")
    return arrays, side - 1


def load_cameras(path: Path, width: int, height: int):
    cameras = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("Camera export must contain at least one camera")

    def capture_order(camera):
        return [int(part) if part.isdecimal() else part.casefold()
                for part in re.split(r"(\d+)", camera["img_name"])]

    views, intrinsics, names = [], [], []
    for camera in sorted(cameras, key=capture_order):
        rotation = np.asarray(camera["rotation"], dtype=np.float64)
        position = np.asarray(camera["position"], dtype=np.float64)
        if (rotation.shape != (3, 3) or position.shape != (3,)
                or not np.isfinite(position).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
            raise ValueError(f"Invalid camera-to-world pose for {camera['img_name']}")
        source_width, source_height = camera["width"], camera["height"]
        fx, fy = camera["fx"], camera["fy"]
        if not all(math.isfinite(value) and value > 0 for value in (source_width, source_height, fx, fy)):
            raise ValueError(f"Invalid camera intrinsics for {camera['img_name']}")
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rotation.T
        view[:3, 3] = -(rotation.T @ position)
        views.append(view)
        intrinsics.append([[fx * width / source_width, 0, width / 2],
                           [0, fy * height / source_height, height / 2],
                           [0, 0, 1]])
        names.append(camera["img_name"])
    return np.stack(views), np.asarray(intrinsics, dtype=np.float32), names
