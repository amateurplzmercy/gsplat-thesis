"""Compare causal fixed and motion-adaptive prefetch against ordinary LRU."""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .render_baseline import timing_stats
from .stream_blocks import MIB, Replay, write_report


def angle(rotation):
    return float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))


def predict(replay, history, calibration, policy, vectorized=False, short_prediction=None):
    if policy == "lru" or len(history) < 2:
        return [], 0
    poses = np.linalg.inv(np.asarray(history, dtype=np.float64))
    before, current = poses[-2:]
    velocity = current[:3, 3] - before[:3, 3]
    step = before[:3, :3].T @ current[:3, :3]
    short = policy == "guarded" if short_prediction is None else short_prediction
    horizon = 1 if short else 4
    if policy in ("adaptive", "guarded") and len(poses) == 3:
        old = poses[0]
        old_velocity = before[:3, 3] - old[:3, 3]
        old_step = old[:3, :3].T @ before[:3, :3]
        move_error = np.linalg.norm(velocity - old_velocity) / max(
            np.linalg.norm(velocity), np.linalg.norm(old_velocity), 1e-4)
        turn_error = angle(old_step.T @ step) / max(angle(step), angle(old_step), np.radians(0.5))
        error = max(move_error, turn_error)
        if short:
            horizon = 0 if error >= 0.7 else 1 if error >= 0.2 else 2
        else:
            horizon = 0 if error >= 0.7 else 2 if error >= 0.2 else 6
    if horizon == 0:
        return [], 0
    views = np.repeat(np.eye(4)[None], horizon, axis=0)
    orientation = current[:3, :3]
    for view in views:
        orientation = orientation @ step
        view[:3, :3] = orientation.T
    if vectorized:
        centers = current[:3, 3] + np.arange(1, horizon + 1)[:, None] * velocity
        views[:, :3, 3] = -(views[:, :3, :3] @ centers[:, :, None]).squeeze(-1)
        forecasts = replay.blocks.visible_many(views, calibration, replay.width, replay.height)
    else:
        for offset, view in enumerate(views, 1):
            center = current[:3, 3] + offset * velocity
            view[:3, 3] = -(view[:3, :3] @ center)
        forecasts = [replay.blocks.visible(view, calibration, replay.width, replay.height) for view in views]
    candidates = {}
    for offset, predicted in enumerate(forecasts, 1):
        if short and offset > 1:
            sustained = set(predicted)
            candidates = dict.fromkeys(block for block in candidates if block in sustained)
        else:
            candidates.update(dict.fromkeys(predicted))
    return list(candidates), horizon


class AsyncUpload:
    def __init__(self, blocks, quota, batch=False):
        self.blocks = blocks
        self.staging = torch.empty(quota // 4, dtype=torch.float32, pin_memory=True)
        self.stream = torch.cuda.Stream()
        self.batch = batch
        with torch.cuda.stream(self.stream):
            self.device_stage = torch.empty(quota // 4, dtype=torch.float32, device="cuda") if batch else None
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.offset = 0

    def stage(self, block):
        cpu = self.blocks.cpu_block(block)
        start = self.offset
        pinned = self.staging[self.offset:self.offset + cpu.numel()]
        self.offset += cpu.numel()
        begin = time.perf_counter()
        pinned.copy_(cpu)
        self.stage_ms += (time.perf_counter() - begin) * 1000
        return start, self.offset, pinned

    def upload_many(self, selected):
        if self.batch:
            spans = {block: self.stage(block)[:2] for block in selected}
            self.device_stage[:self.offset].copy_(self.staging[:self.offset], non_blocking=True)
            # Each resident block owns its allocation so eviction releases its
            # storage independently of the other blocks in this upload.
            return {block: self.blocks.fields(self.device_stage[start:end].clone(),
                                             self.blocks.blocks[block]["count"])
                    for block, (start, end) in spans.items()}
        uploaded = {}
        for block in selected:
            _, _, pinned = self.stage(block)
            uploaded[block] = self.blocks.fields(pinned.to("cuda", non_blocking=True),
                                                self.blocks.blocks[block]["count"])
        return uploaded

    def schedule(self, cache, candidates, required, quota):
        self.offset = 0
        self.stage_ms = 0.0
        with torch.cuda.stream(self.stream):
            self.begin.record()
            admission = cache.admit_prefetch(candidates, required, quota, self.upload_many)
            self.end.record()
        return admission


def replay_once(replay, policy, uploader, quota, frames, observe=None, trace=None,
                launch_order="prefetch-first", vectorized=False, demand_staging=None,
                pack_buffer=None, fault_counter=None, short_prediction=None, spare_admission=None):
    cache = replay.cache
    cache.clear()
    history = deque(maxlen=3)
    unused = set()
    records = []
    smallest_block = int(replay.blocks.sizes.min())
    spare_only = policy == "guarded" if spare_admission is None else spare_admission
    anchor, demand_end, pack_end, render_begin, render_end = [
        torch.cuda.Event(enable_timing=True) for _ in range(5)]

    def submit_render(inputs, frame):
        render_begin.record()
        if trace:
            trace.start_render()
        start = time.perf_counter()
        outputs = replay.render(inputs, frame)
        elapsed = (time.perf_counter() - start) * 1000
        if trace:
            trace.end_render()
        render_end.record()
        return outputs, elapsed

    for frame in range(frames):
        if trace:
            trace.start_frame()
        started = time.perf_counter()
        required = replay.select(frame)
        selected_at = time.perf_counter()
        useful = unused.intersection(required)
        unused.difference_update(useful)
        if demand_staging is not None:
            demand_staging.begin_frame()
        anchor.record()
        faults_before = fault_counter() if fault_counter is not None else None
        demand = cache.prepare(required)
        demand_faults = (fault_counter() - faults_before) if faults_before is not None else None
        demand_end.record()
        demanded_at = time.perf_counter()
        inputs = cache.assemble(required, pack_buffer)
        pack_end.record()
        packed_at = time.perf_counter()
        if launch_order == "render-first":
            (image, alpha, meta), render_submit_ms = submit_render(inputs, frame)
        dropped = unused.difference(cache.entries)
        unused.difference_update(dropped)
        history.append(replay.views[frame])
        selected, transferred, horizon = [], 0, 0
        prefetch_evicted = prefetch_evicted_bytes = 0
        copy_submitted = False
        predict_ms = prefetch_submit_ms = 0.0
        available = min(quota, cache.budget - cache.bytes) if spare_only else quota
        if uploader is not None and frame + 1 < frames and available >= smallest_block:
            begin = time.perf_counter()
            candidates, horizon = predict(replay, history, replay.intrinsics[frame], policy, vectorized,
                                          short_prediction)
            predict_ms = (time.perf_counter() - begin) * 1000
            if any(block not in cache.entries for block in candidates):
                begin = time.perf_counter()
                selected, transferred, prefetch_evicted, prefetch_evicted_bytes = uploader.schedule(
                    cache, candidates, required, available)
                prefetch_submit_ms = (time.perf_counter() - begin) * 1000
                copy_submitted = True
                dropped.update(unused.difference(cache.entries))
                unused.intersection_update(cache.entries)
                unused.update(selected)
        copy_ready = uploader.end.query() if transferred and launch_order == "prefetch-first" else False
        if launch_order == "prefetch-first":
            (image, alpha, meta), render_submit_ms = submit_render(inputs, frame)
        wait_begin = time.perf_counter()
        render_end.synchronize()
        render_ready_at = time.perf_counter()
        if copy_submitted:
            # End the frame after both streams have finished. Pinned staging and
            # evicted storage can be reused safely on the next frame.
            uploader.end.synchronize()
        ready_at = time.perf_counter()
        wall = (ready_at - started) * 1000
        if trace:
            trace.end_frame()
        overlap = 0.0
        if transferred:
            overlap = max(0.0, min(anchor.elapsed_time(uploader.end), anchor.elapsed_time(render_end))
                          - max(anchor.elapsed_time(uploader.begin), anchor.elapsed_time(render_begin)))
        records.append({"frame": frame, "camera": replay.names[frame], "wall_ms": wall,
                        "start_perf_ns": round(started * 1e9),
                        "demand_page_faults": demand_faults,
                        "demand_host_stage_ms": demand_staging.stage_ms if demand_staging else 0.0,
                        "demand_staging_reuse_wait_ms": demand_staging.reuse_wait_ms if demand_staging else 0.0,
                        "select_cpu_ms": (selected_at - started) * 1000,
                        "demand_submit_cpu_ms": (demanded_at - selected_at) * 1000,
                        "demand_window_ms": anchor.elapsed_time(demand_end),
                        "packing_submit_cpu_ms": (packed_at - demanded_at) * 1000,
                        "packing_window_ms": demand_end.elapsed_time(pack_end),
                        "predict_cpu_ms": predict_ms,
                        "prefetch_submit_cpu_ms": prefetch_submit_ms,
                        "prefetch_stage_cpu_ms": uploader.stage_ms if copy_submitted else 0.0,
                        "prefetch_copy_calls": (1 if uploader.batch else len(selected)) if transferred else 0,
                        "prefetch_window_ms": uploader.begin.elapsed_time(uploader.end) if transferred else 0.0,
                        "prefetch_done_before_render_submit": int(copy_ready),
                        "render_submit_cpu_ms": render_submit_ms,
                        "render_window_ms": render_begin.elapsed_time(render_end),
                        "render_wait_cpu_ms": (render_ready_at - wait_begin) * 1000,
                        "prefetch_tail_wait_cpu_ms": (ready_at - render_ready_at) * 1000,
                        "projected_gaussians": meta["gaussian_ids"].numel(), "horizon": horizon,
                        "prefetch_bytes": transferred,
                        "prefetch_d2d_logical_bytes": transferred if uploader and uploader.batch else 0,
                        "prefetch_evicted_blocks": prefetch_evicted,
                        "prefetch_evicted_bytes": prefetch_evicted_bytes,
                        "prefetch_used_bytes": int(replay.blocks.sizes[list(useful)].sum()),
                        "prefetch_wasted_bytes": int(replay.blocks.sizes[list(dropped)].sum()),
                        "prefetch_render_window_overlap_ms": overlap,
                        **{**demand, "resident_bytes": cache.bytes}})
        if observe is not None:
            observe(frame, image)
        del inputs, image, alpha, meta
    records[-1]["prefetch_wasted_bytes"] += int(replay.blocks.sizes[list(unused)].sum())
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--budget-mib", type=int, required=True)
    parser.add_argument("--policy", choices=("lru", "fixed", "adaptive", "guarded"), required=True)
    parser.add_argument("--quota-mib", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-trace", action="store_true", help="Capture one separate CUPTI replay")
    parser.add_argument("--launch-order", choices=("prefetch-first", "render-first"),
                        default="prefetch-first")
    parser.add_argument("--batch-upload", action="store_true")
    parser.add_argument("--vectorized", action="store_true")
    args = parser.parse_args()
    if min(args.budget_mib, args.quota_mib, args.repeats) < 1:
        parser.error("Budget, quota and repeats must be positive")
    if args.device_trace and args.policy == "lru":
        parser.error("Device trace requires a prefetch policy")
    with torch.inference_mode():
        replay = Replay(args.blocks, args.trajectory, args.budget_mib * MIB)
        working_set = replay.feasibility()
        if working_set["over_budget_frames"] or len(replay.names) < 2:
            raise SystemExit("Choose a budget that fits all frames and a trajectory of at least two frames")
        quota = args.quota_mib * MIB
        uploader = AsyncUpload(replay.blocks, quota, batch=args.batch_upload) if args.policy != "lru" else None
        replay_options = {"launch_order": args.launch_order, "vectorized": args.vectorized}
        replay_once(replay, args.policy, uploader, quota, min(len(replay.names), 60),
                    **replay_options)
        replay.cache.clear()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for repeat in range(1, args.repeats + 1):
            records = replay_once(replay, args.policy, uploader, quota, len(replay.names),
                                  **replay_options)
            for record in records:
                record["repeat"] = repeat
            rows.extend(records)
            print(f"{args.policy} pass {repeat}: p95 {np.percentile([r['wall_ms'] for r in records], 95):.2f} ms",
                  flush=True)
        warm = [row for row in rows if row["frame"] > 0]
        sums = {key: sum(row[key] for row in rows) for key in
                ("upload_bytes", "prefetch_bytes", "prefetch_used_bytes", "prefetch_wasted_bytes")}
        if sums["prefetch_bytes"] != sums["prefetch_used_bytes"] + sums["prefetch_wasted_bytes"]:
            raise RuntimeError("Speculative transfer accounting mismatch")
        report = {"status": "passed", "policy": args.policy, "frames": len(rows),
                  "trajectory": args.trajectory.as_posix(), "budget_mib": args.budget_mib,
                  "launch_order": args.launch_order,
                  "batch_upload": args.batch_upload, "vectorized": args.vectorized,
                  "gpu_upload_scratch_bytes": quota if args.batch_upload and uploader else 0,
                  "quota_mib": args.quota_mib, "peak_block_bytes": max(row["resident_bytes"] for row in rows),
                  "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "warm_wall_ms": timing_stats([row["wall_ms"] for row in warm]),
                  "warm_p99_ms": float(np.percentile([row["wall_ms"] for row in warm], 99)),
                  "warm_demand_window_ms": timing_stats([row["demand_window_ms"] for row in warm]),
                  "warm_miss_frames": sum(row["upload_bytes"] > 0 for row in warm),
                  "warm_demand_bytes": sum(row["upload_bytes"] for row in warm),
                  "timing_scope": "Closed-loop replay; wall time includes rendering and prefetch completion.",
                  **sums}
        report["warm_stages_ms"] = {
            key: timing_stats([row[key] for row in warm])
            for key in warm[0] if key.endswith("_ms")
        }
        write_report(args.output, report, rows)
        if args.device_trace:
            from .cuda_activity import ActivityTrace

            with ActivityTrace(uploader.stream) as trace:
                traced = replay_once(replay, args.policy, uploader, quota, len(replay.names), trace=trace,
                                     **replay_options)
            trace.write(args.output, traced)


if __name__ == "__main__":
    main()
