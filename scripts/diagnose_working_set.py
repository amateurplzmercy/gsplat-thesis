"""Measure projection coverage, block granularity cost and view selection cost."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .graphdeco_scene import load_gaussians
from .prepare_workloads import rotation_angle
from .render_settings import render_scene
from .spatial_blocks import Blocks, spatial_leaves
from .stream_blocks import BlockCache, MIB


def owner_for_layout(groups, means, leaf_size):
    owner = np.empty(len(means), dtype=np.int32)
    counts = []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group).astype(np.int32)
        for ids in spatial_leaves(means, members, leaf_size):
            owner[ids] = len(counts)
            counts.append(len(ids))
    return owner, np.asarray(counts)


def compare_lru_timings(rows, timings, budget):
    by_frame = defaultdict(list)
    with timings.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            if record["policy"] == "lru" and int(record["budget_mib"]) == budget:
                by_frame[int(record["frame"])].append(record)
    for row in rows:
        samples = by_frame[row["frame"]]
        if not samples or any(sample["camera"] != row["camera"] or
                              int(sample["upload_bytes"]) != row["selected_miss_bytes"]
                              for sample in samples):
            raise ValueError(f"Timing samples do not match the simulated LRU at frame {row['frame']}")
        row.update({f"lru_median_{field}": float(np.median([float(s[field]) for s in samples]))
                    for field in ("wall_ms", "demand_submit_cpu_ms", "demand_window_ms",
                                  "select_cpu_ms", "packing_window_ms", "render_window_ms")})
        row["lru_slow_frames"] = sum(float(s["wall_ms"]) > 33.3 for s in samples)
        row["lru_timed_frames"] = len(samples)

    warm = rows[1:]
    fields = ("translation", "rotation_deg", "selected_adjacent_new_bytes", "selected_miss_bytes")
    wall = [r["lru_median_wall_ms"] for r in warm]
    correlations = {field: float(np.corrcoef([r[field] for r in warm], wall)[0, 1])
                    for field in fields if np.ptp([r[field] for r in warm]) > 0}
    bins = []
    for low, high in ((0, 16), (16, 64), (64, 96), (96, None)):
        subset = [r for r in warm if r["selected_miss_bytes"] >= low * MIB and
                  (high is None or r["selected_miss_bytes"] < high * MIB)]
        if subset:
            bins.append({"range_mib": [low, high], "camera_frames": len(subset),
                         "median_wall_ms": float(np.median([r["lru_median_wall_ms"] for r in subset])),
                         "median_demand_submit_ms": float(np.median(
                             [r["lru_median_demand_submit_cpu_ms"] for r in subset])),
                         "slow_frames": sum(r["lru_slow_frames"] for r in subset),
                         "timed_frames": sum(r["lru_timed_frames"] for r in subset)})
    return {"source": timings.as_posix(), "pearson_r_vs_median_wall": correlations, "by_demand_mib": bins}


def measure(scene, block_folder, trajectory, output, budget=None, timings=None):
    index = json.loads((block_folder / "index.json").read_text(encoding="utf-8"))
    if json.loads((scene / "source.json").read_text(encoding="utf-8")) != index["source"]:
        raise ValueError("The block index and reference scene must use the same model")
    original_ids = np.load(block_folder / "original_ids.npy", allow_pickle=False)
    arrays, degree = load_gaussians(scene / "point_cloud.ply")
    count = len(arrays["means"])
    counts = np.array([block["count"] for block in index["blocks"]])
    owner = np.empty(count, dtype=np.int32)
    owner[original_ids] = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    layouts = {"oracle_current_count": (owner, counts)}
    groups = np.digitize(arrays["scales"].max(axis=1), index.get("scale_bands", []))
    for leaf_size in (1024, 512, 256):
        layouts[f"oracle_{leaf_size}_count"] = owner_for_layout(groups, arrays["means"], leaf_size)
    bounds = Blocks(block_folder)
    trajectory = json.loads(trajectory.read_text(encoding="utf-8"))
    views = np.asarray(trajectory["viewmats"], dtype=np.float32)
    intrinsics = np.asarray(trajectory["intrinsics"], dtype=np.float32)
    width, height = trajectory["resolution"]
    views_gpu = torch.from_numpy(views).to("cuda")
    intrinsics_gpu = torch.from_numpy(intrinsics).to("cuda")
    inputs = {key: torch.from_numpy(value).to("cuda") for key, value in arrays.items()}
    del arrays
    bytes_per_gaussian = 4 * index["floats_per_gaussian"]
    request_sizes = {"selected": counts * bytes_per_gaussian}
    for key, (_, sizes) in layouts.items():
        request_sizes[key.removesuffix("_count")] = sizes * bytes_per_gaussian
    caches = {key: BlockCache(SimpleNamespace(sizes=sizes), budget * MIB, upload=lambda _: None)
              for key, sizes in request_sizes.items()} if budget else {}
    poses = np.linalg.inv(views.astype(np.float64))
    centers, rotations = poses[:, :3, 3], poses[:, :3, :3]
    translations = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    turns = np.degrees(rotation_angle(rotations[:-1].transpose(0, 2, 1) @ rotations[1:]))
    previous = {key: set() for key in request_sizes}

    rows = []
    for frame, name in enumerate(trajectory["camera_names"]):
        _, _, meta = render_scene(inputs, views_gpu[frame:frame + 1], intrinsics_gpu[frame:frame + 1],
                                  width, height, degree)
        ids = meta["gaussian_ids"].cpu().numpy()
        selected = bounds.visible(views[frame], intrinsics[frame], width, height)
        row = {"frame": frame, "camera": name, "projected_count": len(ids),
               "translation": float(translations[frame-1]) if frame else 0.0,
               "rotation_deg": float(turns[frame-1]) if frame else 0.0,
               "selected_count": int(counts[selected].sum())}
        requests = {"selected": selected}
        for key, (layout, sizes) in layouts.items():
            touched = np.flatnonzero(np.bincount(layout[ids], minlength=len(sizes)))
            requests[key.removesuffix("_count")] = touched.tolist()
            row[key] = int(sizes[touched].sum())
        if set(requests["oracle_current"]) - set(selected):
            raise RuntimeError(f"Selector excludes a projected block at frame {frame}")
        for key, blocks in requests.items():
            required = set(blocks)
            sizes = request_sizes[key]
            row[f"{key}_required_bytes"] = int(sizes[blocks].sum())
            row[f"{key}_adjacent_new_bytes"] = int(sizes[list(required - previous[key])].sum())
            row[f"{key}_adjacent_shared_bytes"] = int(sizes[list(required & previous[key])].sum())
            if key in caches:
                row[f"{key}_miss_bytes"] = caches[key].prepare(blocks)["upload_bytes"]
            previous[key] = required
        rows.append(row)
    summary = {"scene": scene.name, "gaussians": count, "frames": len(rows),
               "resolution": [width, height], "current_blocks": len(counts),
               "current_leaf_size": index["leaf_size"], "budget_mib": budget,
               "bytes_per_gaussian": bytes_per_gaussian, "percent_of_full_payload": {}}
    if timings:
        summary["observed_lru"] = compare_lru_timings(rows, timings, budget)
    if budget:
        summary["warm_miss_mib"] = {}
        for key in request_sizes:
            miss = np.array([row[f"{key}_miss_bytes"] / MIB for row in rows[1:]])
            summary["warm_miss_mib"][key] = {"total": float(miss.sum()), "p95": float(np.percentile(miss, 95)),
                                            "max": float(miss.max())}
    for key in rows[0]:
        if not key.endswith("_count"):
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64) * (100 / count)
        percentiles = np.percentile(values, [0, 50, 95, 100]).tolist()
        summary["percent_of_full_payload"][key] = dict(zip(("min", "median", "p95", "max"), percentiles))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = json.dumps(summary, indent=2) + "\n"
    (output / "summary.json").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("data/train"))
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, help="Simulate LRU misses under this budget in MiB")
    parser.add_argument("--timings", type=Path, help="Join measured LRU frame logs at that budget")
    args = parser.parse_args()
    if (args.budget is not None and args.budget <= 0) or (args.timings and args.budget is None):
        parser.error("Use a positive --budget when simulating or joining timings")
    with torch.inference_mode():
        measure(args.scene, args.blocks, args.trajectory, args.output, args.budget, args.timings)


if __name__ == "__main__":
    main()
