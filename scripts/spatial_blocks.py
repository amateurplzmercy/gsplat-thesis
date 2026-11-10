"""Build and query spatial blocks for the pinned gsplat pinhole renderer."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .graphdeco_scene import load_gaussians
from .render_settings import FAR, NEAR


FIELDS = ("means", "quats", "scales", "opacities", "colors")
SUPPORT = 4.4
PIXEL_PAD = 4.0


def spatial_leaves(means, ids, leaf_size):
    if len(ids) <= leaf_size:
        yield np.sort(ids)
        return
    points = means[ids]
    axis = np.argmax(np.ptp(points, axis=0))
    midpoint = len(ids) // 2
    ordered = ids[np.argpartition(points[:, axis], midpoint)]
    yield from spatial_leaves(means, ordered[:midpoint], leaf_size)
    yield from spatial_leaves(means, ordered[midpoint:], leaf_size)


def build(scene, destination, leaf_size, scale_bands):
    if destination.exists():
        raise FileExistsError(f"Choose a new block directory: {destination}")
    source = json.loads((scene / "source.json").read_text(encoding="utf-8"))
    arrays, degree = load_gaussians(scene / "point_cloud.ply")
    means = arrays["means"]
    count = len(means)
    shapes = {key: list(arrays[key].shape[1:]) for key in FIELDS}
    widths = {key: int(np.prod(shapes[key])) for key in FIELDS}
    stride = sum(widths.values())
    # The clamped pinhole Jacobian and 3.33 sigma screen footprint imply a
    # 3.33 * 1.3 bound along normalized frustum plane normals (centered camera).
    max_scale = np.max(arrays["scales"], axis=1)
    radius = SUPPORT * max_scale
    lower, upper = means - radius[:, None], means + radius[:, None]
    staged = destination.with_name(destination.name + ".partial")
    staged.mkdir(parents=True, exist_ok=False)
    payload = np.lib.format.open_memmap(staged / "payload.npy", mode="w+",
                                       dtype=np.float32, shape=(count * stride,))
    original_ids = np.empty(count, dtype=np.int32)
    blocks = []
    first = 0
    groups = np.digitize(max_scale, scale_bands)
    leaves = (ids for group in np.unique(groups)
              for ids in spatial_leaves(means, np.flatnonzero(groups == group).astype(np.int32), leaf_size))
    for ids in leaves:
        offset = first * stride
        original_ids[first:first + len(ids)] = ids
        for key in FIELDS:
            size = len(ids) * widths[key]
            payload[offset:offset + size] = arrays[key][ids].reshape(-1)
            offset += size
        blocks.append({"first": first, "count": len(ids),
                       "minimum": lower[ids].min(axis=0).tolist(),
                       "maximum": upper[ids].max(axis=0).tolist()})
        first += len(ids)
    payload.flush()
    del payload
    np.save(staged / "original_ids.npy", original_ids, allow_pickle=False)
    metadata = {"version": 1, "source": source, "leaf_size": leaf_size,
                "scale_bands": scale_bands, "gaussians": count, "sh_degree": degree, "shapes": shapes,
                "floats_per_gaussian": stride, "support_sigma": SUPPORT,
                "pixel_pad": PIXEL_PAD, "blocks": blocks}
    (staged / "index.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    staged.rename(destination)
    print(f"Packed {count:,} Gaussians in {len(blocks)} blocks, {count * stride * 4 / 2**20:.1f} MiB")


class Blocks:
    def __init__(self, folder):
        self.index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        if (self.index["version"] != 1 or self.index["support_sigma"] != SUPPORT
                or self.index["pixel_pad"] != PIXEL_PAD):
            raise ValueError("Unsupported block index or footprint constants")
        self.blocks = self.index["blocks"]
        self.stride = self.index["floats_per_gaussian"]
        self.shapes = self.index["shapes"]
        self.widths = [int(np.prod(self.shapes[key])) for key in FIELDS]
        payload = np.load(folder / "payload.npy", allow_pickle=False)
        if payload.dtype != np.float32 or payload.shape != (self.index["gaussians"] * self.stride,):
            raise ValueError("Block payload does not match index")
        self.payload = torch.from_numpy(payload)
        self.sizes = np.array([block["count"] * self.stride * 4 for block in self.blocks], dtype=np.int64)
        minimum = np.array([block["minimum"] for block in self.blocks], dtype=np.float64)
        maximum = np.array([block["maximum"] for block in self.blocks], dtype=np.float64)
        self.centers = (minimum + maximum) / 2
        self.extents = (maximum - minimum) / 2

    def camera_planes(self, intrinsics, width, height):
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        if cx != width / 2 or cy != height / 2 or min(fx, fy) <= 0:
            raise ValueError("Block footprint requires centered pinhole intrinsics")
        x = (cx + PIXEL_PAD) / fx
        y = (cy + PIXEL_PAD) / fy
        return np.array([[1, 0, x], [-1, 0, x], [0, 1, y], [0, -1, y],
                         [0, 0, 1], [0, 0, -1]], dtype=np.float64)

    def visible(self, view, intrinsics, width, height):
        camera_normals = self.camera_planes(intrinsics, width, height)
        normals = camera_normals @ view[:3, :3]
        offsets = camera_normals @ view[:3, 3]
        offsets[4] -= NEAR
        offsets[5] += FAR
        furthest = self.centers @ normals.T + self.extents @ np.abs(normals).T + offsets
        return np.flatnonzero((furthest >= -1e-4).all(axis=1)).tolist()

    def visible_many(self, views, intrinsics, width, height):
        camera_normals = self.camera_planes(intrinsics, width, height)
        normals = camera_normals @ views[:, :3, :3]
        offsets = (camera_normals @ views[:, :3, 3, None]).squeeze(-1)
        offsets[:, 4] -= NEAR
        offsets[:, 5] += FAR
        furthest = self.centers @ normals.swapaxes(1, 2)
        furthest += self.extents @ np.abs(normals).swapaxes(1, 2)
        furthest += offsets[:, None, :]
        return [np.flatnonzero(mask).tolist() for mask in (furthest >= -1e-4).all(axis=2)]

    def fields(self, flat, count):
        spans = flat.split([count * width for width in self.widths])
        return {key: span.reshape(count, *self.shapes[key]) for key, span in zip(FIELDS, spans)}

    def cpu_block(self, block_id):
        block = self.blocks[block_id]
        first = block["first"] * self.stride
        count = block["count"]
        return self.payload[first:first + count * self.stride]

    def upload(self, block_id):
        flat = self.cpu_block(block_id).to("cuda", non_blocking=True)
        return self.fields(flat, self.blocks[block_id]["count"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leaf-size", type=int, default=4096)
    parser.add_argument("--scale-bands", type=float, nargs="*", default=[])
    args = parser.parse_args()
    if args.leaf_size <= 0:
        parser.error("Leaf size must be positive")
    if args.scale_bands != sorted(set(args.scale_bands)) or any(
            not np.isfinite(value) or value <= 0 for value in args.scale_bands):
        parser.error("Scale bands must be positive and strictly increasing")
    build(args.scene, args.output, args.leaf_size, args.scale_bands)


if __name__ == "__main__":
    main()
