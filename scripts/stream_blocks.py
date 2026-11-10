"""Replay demand loading and byte-budgeted LRU against a fixed scene trajectory."""

import argparse
import csv
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .graphdeco_scene import load_gaussians
from .render_baseline import timing_stats
from .render_settings import render_scene
from .smoke_render import save_png
from .spatial_blocks import Blocks, FIELDS


MIB = 2**20


class BlockCache:
    def __init__(self, blocks, budget, upload=None):
        self.blocks = blocks
        self.budget = budget
        self.upload = blocks.upload if upload is None else upload
        self.entries = OrderedDict()
        self.bytes = 0

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def prepare(self, required):
        sizes = self.blocks.sizes
        required_bytes = int(sizes[required].sum())
        if required_bytes > self.budget:
            raise ValueError(f"Frame needs {required_bytes / MIB:.2f} MiB; budget {self.budget / MIB:.2f} MiB")
        missing = [block for block in required if block not in self.entries]
        transfer = int(sizes[missing].sum())
        evicted, evicted_bytes = self.make_room(required, transfer)
        for block in missing:
            self.entries[block] = self.upload(block)
            self.bytes += int(sizes[block])
        for block in required:
            self.entries.move_to_end(block)
        return {
            "required_blocks": len(required), "required_bytes": required_bytes,
            "hit_blocks": len(required) - len(missing), "hit_bytes": required_bytes - transfer,
            "upload_blocks": len(missing), "upload_bytes": transfer,
            "evicted_blocks": evicted, "evicted_bytes": evicted_bytes, "resident_bytes": self.bytes,
        }

    def make_room(self, protected, incoming_bytes):
        if self.bytes + incoming_bytes <= self.budget:
            return 0, 0
        protected = set(protected)
        before = self.bytes
        evicted = 0
        for block in list(self.entries):
            if block in protected:
                continue
            del self.entries[block]
            self.bytes -= int(self.blocks.sizes[block])
            evicted += 1
            if self.bytes + incoming_bytes <= self.budget:
                return evicted, before - self.bytes
        raise RuntimeError("Cache admission failed despite feasible working set")

    def admit_prefetch(self, candidates, required, limit, upload_many):
        available = min(limit, self.budget - int(self.blocks.sizes[required].sum()))
        selected = []
        used = 0
        for block in candidates:
            size = int(self.blocks.sizes[block])
            if block not in self.entries and used + size <= available:
                selected.append(block)
                used += size
        evicted = evicted_bytes = 0
        if selected:
            evicted, evicted_bytes = self.make_room(required, used)
            for block, fields in upload_many(selected).items():
                self.entries[block] = fields
                self.bytes += int(self.blocks.sizes[block])
                self.entries.move_to_end(block, last=False)
        return selected, used, evicted, evicted_bytes

    def assemble(self, required, buffer=None):
        entries = [self.entries[block] for block in required]
        if buffer is not None:
            count = sum(entry["means"].shape[0] for entry in entries)
            if count > buffer["means"].shape[0]:
                raise ValueError("Frame exceeds preallocated packing capacity")
            return {field: torch.cat([entry[field] for entry in entries], out=buffer[field][:count])
                    for field in FIELDS}
        return {field: torch.cat([entry[field] for entry in entries]) for field in FIELDS}


class Replay:
    def __init__(self, folder, trajectory, budget):
        self.blocks = Blocks(folder)
        self.cache = BlockCache(self.blocks, budget)
        self.trajectory = json.loads(trajectory.read_text(encoding="utf-8"))
        self.views = np.asarray(self.trajectory["viewmats"], dtype=np.float32)
        self.intrinsics = np.asarray(self.trajectory["intrinsics"], dtype=np.float32)
        self.names = self.trajectory["camera_names"]
        self.width, self.height = self.trajectory["resolution"]
        self.view_gpu = torch.from_numpy(self.views).to("cuda")
        self.intrinsics_gpu = torch.from_numpy(self.intrinsics).to("cuda")
        self.degree = self.blocks.index["sh_degree"]
        self.payload = int(self.blocks.sizes.sum())

    def select(self, frame):
        required = self.blocks.visible(self.views[frame], self.intrinsics[frame], self.width, self.height)
        if not required:
            raise ValueError(f"No blocks selected at camera {self.names[frame]}")
        return required

    def render(self, inputs, frame):
        return render_scene(inputs, self.view_gpu[frame:frame + 1], self.intrinsics_gpu[frame:frame + 1],
                            self.width, self.height, self.degree)

    def feasibility(self):
        demands = np.array([self.blocks.sizes[self.select(frame)].sum() for frame in range(len(self.names))])
        return {"min": int(demands.min()), "median": float(np.median(demands)), "max": int(demands.max()),
                "over_budget_frames": int((demands > self.cache.budget).sum()),
                "over_provisional_budget_frames": {str(fraction): int((demands > self.payload * fraction).sum())
                                                   for fraction in (0.25, 0.50, 0.75)}}


def audit(replay, folder, scene, output):
    if json.loads((scene / "source.json").read_text(encoding="utf-8")) != replay.blocks.index["source"]:
        raise ValueError("Block index and full scene come from different sources")
    arrays, degree = load_gaussians(scene / "point_cloud.ply")
    if degree != replay.degree or len(arrays["means"]) != replay.blocks.index["gaussians"]:
        raise ValueError("Block index and full scene have different layouts")
    reference = {key: torch.from_numpy(array).to("cuda") for key, array in arrays.items()}
    del arrays
    original_ids = np.load(folder / "original_ids.npy", allow_pickle=False)
    count = len(reference["means"])
    if original_ids.shape != (count,) or not np.array_equal(np.sort(original_ids), np.arange(count)):
        raise ValueError("Block index must cover each original Gaussian exactly once")
    permutation = torch.from_numpy(original_ids).to(device="cuda", dtype=torch.int64)
    ordered_reference = {key: value.index_select(0, permutation) for key, value in reference.items()}
    del permutation
    rows = []
    worst_mse = -1
    for frame, name in enumerate(replay.names):
        required = replay.select(frame)
        cache = replay.cache.prepare(required)
        inputs = replay.cache.assemble(required)
        full, _, full_meta = replay.render(reference, frame)
        streamed, _, meta = replay.render(inputs, frame)
        ordered, _, _ = replay.render(ordered_reference, frame)
        block_ids = np.concatenate([
            original_ids[block["first"]:block["first"] + block["count"]]
            for block in (replay.blocks.blocks[block_id] for block_id in required)])
        full_mask = np.zeros(count, dtype=bool)
        streamed_mask = np.zeros_like(full_mask)
        full_mask[full_meta["gaussian_ids"].cpu().numpy()] = True
        streamed_mask[block_ids[meta["gaussian_ids"].cpu().numpy()]] = True
        delta = (full.clamp(0, 1) - streamed.clamp(0, 1)).abs()
        ordered_delta = (ordered.clamp(0, 1) - streamed.clamp(0, 1)).abs().max()
        mse, max_error, ordered_error = torch.stack(
            (delta.square().mean(), delta.max(), ordered_delta)).cpu().tolist()
        if not all(math.isfinite(value) for value in (mse, max_error, ordered_error)):
            raise RuntimeError(f"Non-finite output at camera {name}")
        rows.append({"frame": frame, "camera": name,
                     "missing_projected": int((full_mask & ~streamed_mask).sum()),
                     "extra_projected": int((streamed_mask & ~full_mask).sum()),
                     "mse_rgb": mse, "max_error_rgb": max_error,
                     "same_order_max_error_rgb": ordered_error,
                     "psnr_db": -10 * math.log10(mse) if mse else None, **cache})
        if mse > worst_mse:
            worst_mse = mse
            worst_frame = frame
            worst_pair = torch.cat((full[0], streamed[0]), dim=1)
            worst_pair = (worst_pair.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        del inputs, full, streamed, ordered, full_meta, meta, delta
        if (frame + 1) % 100 == 0:
            print(f"Quality audit: {frame + 1}/{len(replay.names)}", flush=True)
    missing = sum(row["missing_projected"] for row in rows)
    extra = sum(row["extra_projected"] for row in rows)
    psnr = [row["psnr_db"] for row in rows if row["psnr_db"] is not None]
    report = {"status": "passed" if missing == extra == 0 else "projection_mismatch",
              "frames": len(rows), "missing_projected_total": missing, "extra_projected_total": extra,
              "identical_rgb_frames": len(rows) - len(psnr),
              "min_finite_psnr_db": min(psnr) if psnr else None,
              "max_absolute_rgb_error": max(row["max_error_rgb"] for row in rows),
              "same_order_max_absolute_rgb_error": max(row["same_order_max_error_rgb"] for row in rows),
              "same_order_identical_rgb_frames": sum(row["same_order_max_error_rgb"] == 0 for row in rows),
              "worst_mse_frame": worst_frame, "budget_bytes": replay.cache.budget}
    output.mkdir(parents=True, exist_ok=True)
    save_png(output / "worst_pair.png", worst_pair.shape[1], worst_pair.shape[0], worst_pair.tobytes())
    write_report(output, report, rows)
    if missing or extra:
        raise RuntimeError("Projection audit found a coverage mismatch")


def benchmark(replay, policy, output, warmup, repeats, working_set):
    cache = replay.cache
    rows = []
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for run in range(repeats + 1):
        cache.clear()
        if run == 1:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        frames = np.linspace(0, len(replay.names) - 1, warmup, dtype=int) if run == 0 else range(len(replay.names))
        for frame in frames:
            frame_start = time.perf_counter()
            required = replay.select(frame)
            select_ms = (time.perf_counter() - frame_start) * 1000
            events[0].record()
            counters = cache.prepare(required)
            events[1].record()
            inputs = cache.assemble(required)
            events[2].record()
            image, alpha, meta = replay.render(inputs, frame)
            events[3].record()
            events[3].synchronize()
            wall_ms = (time.perf_counter() - frame_start) * 1000
            if run:
                rows.append({"repeat": run, "frame": frame, "camera": replay.names[frame],
                             "wall_ms": wall_ms, "select_ms": select_ms,
                             "admission_window_ms": events[0].elapsed_time(events[1]),
                             "packing_window_ms": events[1].elapsed_time(events[2]),
                             "gpu_render_ms": events[2].elapsed_time(events[3]),
                             "projected_gaussians": meta["gaussian_ids"].numel(), **counters})
            del inputs, image, alpha, meta
            if policy == "demand":
                cache.clear()
        if run:
            values = [row["wall_ms"] for row in rows if row["repeat"] == run]
            print(f"{policy} pass {run}: p95 {np.percentile(values, 95):.2f} ms", flush=True)
    transferred = sum(row["upload_bytes"] for row in rows)
    requested = sum(row["required_bytes"] for row in rows)
    report = {
        "status": "passed", "policy": policy, "resolution": [replay.width, replay.height],
        "sh_degree": replay.degree, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "frames": len(rows), "warmup_frames": warmup, "repeats": repeats,
        "budget_bytes": cache.budget, "full_payload_bytes": replay.payload,
        "working_set_bytes": working_set, "peak_resident_block_bytes": max(row["resident_bytes"] for row in rows),
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "uploaded_bytes": transferred, "requested_bytes": requested,
        "byte_hit_rate": 1 - transferred / requested,
        "block_hit_rate": sum(row["hit_blocks"] for row in rows) / sum(row["required_blocks"] for row in rows),
        "frames_over_33_3ms": sum(row["wall_ms"] > 33.3 for row in rows),
        "timings_ms": {key: timing_stats([row[key] for row in rows]) for key in
                       ("wall_ms", "select_ms", "admission_window_ms", "packing_window_ms", "gpu_render_ms")},
        "timing_scope": "Cache reset at each measured replay; end-to-end frame includes CPU selection, "
                        "upload scheduling, GPU packing and rasterization. Quality audit runs separately.",
    }
    write_report(output, report, rows)


def write_report(output, report, rows):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("audit", "demand", "lru"), required=True)
    parser.add_argument("--budget-mib", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.budget_mib, args.repeats) < 1 or args.warmup < 0:
        parser.error("Use positive budget/repeats and nonnegative warmup")
    with torch.inference_mode():
        replay = Replay(args.blocks, args.trajectory, args.budget_mib * MIB)
        working_set = replay.feasibility()
        print(f"Peak required: {working_set['max'] / MIB:.2f} MiB", flush=True)
        if working_set["over_budget_frames"]:
            raise SystemExit(f"Infeasible budget in {working_set['over_budget_frames']} of {len(replay.names)} frames")
        if args.policy == "audit":
            audit(replay, args.blocks, args.scene, args.output)
        else:
            benchmark(replay, args.policy, args.output, args.warmup, args.repeats, working_set)


if __name__ == "__main__":
    main()
