"""Run host-path diagnostics and position-balanced policy/budget replays."""

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

import torch

from .compare_prefetch import AsyncUpload, replay_once
from .replay_buffers import PinnedDemand, packing_buffer
from .run_prefetch_matrix import ORDERS, summarize
from .stream_blocks import MIB, Replay
from .system_telemetry import Sampler, payload_residency, pretouch, process_memory


BUDGETS = {"continuous": (298, 326, 430), "return": (298, 326, 430), "capture_order": (375,)}
HOST_ARMS = {
    "legacy": (False, False, False),
    "touch": (True, False, False),
    "pinned": (True, True, False),
    "reuse": (True, True, True),
}
SHORT_PIPELINE_ARMS = {
    "single_scalar": {"batch": False, "vectorized": False, "short_prediction": True},
    "single_vector": {"batch": False, "vectorized": True, "short_prediction": True},
    "batch_scalar": {"batch": True, "vectorized": False, "short_prediction": True},
    "batch_vector": {"batch": True, "vectorized": True, "short_prediction": True},
}
ABLATIONS = {
    "pipeline": {
        "original": {"launch_order": "prefetch-first", "batch": False, "vectorized": False},
        "reordered": {"batch": False, "vectorized": False},
        "batched": {"vectorized": False},
        "vectorized": {},
    },
    "admission": {
        "long": {"short_prediction": False, "spare_admission": False},
        "short_only": {"short_prediction": True, "spare_admission": False},
        "spare_only": {"short_prediction": False, "spare_admission": True},
        "guarded": {"short_prediction": True, "spare_admission": True},
    },
    "candidate": {"lru": {}, "fixed": {}, "short_only": {"short_prediction": True}, "guarded": {}},
    "short_pipeline": SHORT_PIPELINE_ARMS,
    "short_pipeline_confirm": {arm: SHORT_PIPELINE_ARMS[arm] for arm in ("single_vector", "batch_vector")},
}
ABLATION_PAIRS = {
    "pipeline": (("reordered", "original"), ("batched", "reordered"), ("vectorized", "batched")),
    "admission": (("short_only", "long"), ("spare_only", "long"),
                  ("guarded", "short_only"), ("guarded", "spare_only")),
    "candidate": (("fixed", "lru"), ("short_only", "lru"), ("short_only", "fixed"),
                  ("short_only", "guarded"), ("guarded", "lru")),
    "short_pipeline": (("single_scalar", "batch_vector"), ("batch_scalar", "single_scalar"),
                       ("batch_vector", "single_vector"), ("single_vector", "single_scalar"),
                       ("batch_vector", "batch_scalar")),
    "short_pipeline_confirm": (("single_vector", "batch_vector"),),
}
SHORT_PIPELINE_TRAFFIC = ("upload_bytes", "prefetch_bytes", "prefetch_used_bytes",
                          "prefetch_wasted_bytes", "resident_bytes", "evicted_bytes",
                          "prefetch_evicted_bytes", "horizon")


def allocator_counts():
    stats = torch.cuda.memory_stats()
    return {key: stats.get(key) for key in ("num_alloc_retries", "num_sync_all_streams", "num_device_alloc")}


def run_track(track, args, folder, sampler):
    budgets = (args.budget,) if args.budget is not None else BUDGETS[track]
    if args.diagnose or args.ablation:
        budgets = budgets[:1]
    comparison = "candidate" if args.candidate_matrix else args.ablation
    short_pipeline = comparison in ("short_pipeline", "short_pipeline_confirm")
    replay = Replay(args.blocks, args.workloads / f"{track}.json", budgets[0] * MIB)
    peak = replay.feasibility()["max"]
    if peak > min(budgets) * MIB:
        raise RuntimeError(f"{track} cannot fit the smallest budget")
    quota = 16 * MIB
    uploader = AsyncUpload(replay.blocks, quota, batch=True)
    single = AsyncUpload(replay.blocks, quota) if comparison == "pipeline" or short_pipeline else None
    staging = PinnedDemand(replay.blocks) if args.diagnose or args.pinned else None
    packed = packing_buffer(replay.blocks, peak) if args.diagnose or args.reuse else None

    ablation_arms = ABLATIONS.get(comparison)
    arms = tuple(HOST_ARMS if args.diagnose else ablation_arms or ORDERS[0])
    if len(arms) == 2:
        orders = (arms, arms[::-1])
    else:
        aliases = dict(zip(ORDERS[0], arms))
        orders = tuple(tuple(aliases[arm] for arm in order) for order in ORDERS)

    def run(policy, arm, budget, frames, flags, probe=False):
        _, use_pinned, use_reuse = flags
        replay.cache.budget = budget * MIB
        replay.cache.upload = staging if use_pinned else replay.blocks.upload
        options = {"launch_order": "render-first", "vectorized": True}
        if ablation_arms:
            options.update(ablation_arms[arm])
        transport = uploader if options.pop("batch", True) else single
        return replay_once(replay, policy, None if policy == "lru" else transport, quota, frames,
                           **options,
                           demand_staging=staging if use_pinned else None,
                           pack_buffer=packed if use_reuse else None,
                           fault_counter=(lambda: process_memory()[0]) if probe else None)

    flags = (True, args.pinned, args.reuse)
    def settings(arm):
        if args.diagnose:
            return "lru", HOST_ARMS[arm]
        if args.ablation == "pipeline":
            return "fixed", flags
        if comparison == "admission" or short_pipeline or arm == "short_only":
            return "adaptive", flags
        return arm, flags

    for arm in arms:
        policy, warm_flags = settings(arm)
        run(policy, arm, budgets[0], min(60, len(replay.names)), warm_flags)
    folder.mkdir(parents=True, exist_ok=True)
    runs = []
    reference = None
    fingerprint_fields = ("required_bytes", "projected_gaussians")
    if short_pipeline:
        fingerprint_fields += SHORT_PIPELINE_TRAFFIC
    with (folder / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = None
        for index in range(args.rounds):
            for position in range(len(arms)):
                # Mix capacities at each position while retaining balanced arm order.
                shift = (index + position) % len(budgets)
                rotated = budgets[shift:] + budgets[:shift]
                for budget_position, budget in enumerate(rotated, 1):
                    arm = orders[(index + budgets.index(budget)) % len(orders)][position]
                    policy, run_flags = settings(arm)
                    label = f"{track}/{index+1}/{budget}/{arm}"
                    before_touch = payload_residency(replay.blocks.payload)
                    start = time.perf_counter()
                    if run_flags[0]:
                        pretouch(replay.blocks.payload)
                    touch_ms = (time.perf_counter() - start) * 1000
                    resident_before = payload_residency(replay.blocks.payload)
                    pre = sampler.sample(label + "/before")
                    allocations = allocator_counts()
                    rows = run(policy, arm, budget, len(replay.names), run_flags, probe=True)
                    post = sampler.sample(label + "/after")
                    final_allocations = allocator_counts()
                    resident_after = payload_residency(replay.blocks.payload)
                    fingerprint = [tuple(row[field] for field in fingerprint_fields) for row in rows]
                    if reference is None:
                        reference = fingerprint
                    if fingerprint != reference or any(row["resident_bytes"] > budget * MIB for row in rows):
                        raise RuntimeError(f"Frame coverage, cache budget or traffic mismatch in {label}")
                    tags = {"round": index + 1, "budget_mib": budget, "policy": arm,
                            "position": position + 1, "budget_position": budget_position}
                    run_record = {
                        **tags, **summarize(rows), "before": pre, "after": post,
                        "pre_touch_ms": touch_ms, "payload_before_touch": before_touch,
                        "payload_before": resident_before, "payload_after": resident_after,
                        "allocator_deltas": {key: final_allocations[key] - value if value is not None else None
                                             for key, value in allocations.items()},
                    }
                    runs.append(run_record)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=[*tags, *rows[0]])
                        writer.writeheader()
                    writer.writerows({**tags, **row} for row in rows)
            stream.flush()
            print(f"{track} round {index+1}/{args.rounds} recorded", flush=True)
    report = {"track": track, "rounds": args.rounds, "budgets_mib": budgets, "order_size": len(arms),
              "diagnostic": args.diagnose, "formal_host_flags": flags if not args.diagnose else None,
              "ablation": comparison, "ablation_contrasts": ABLATION_PAIRS.get(comparison, []),
              "candidate_matrix": args.candidate_matrix,
              "frame_count": len(replay.names), "runs": runs}
    (folder / "runs.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnose", action="store_true")
    mode.add_argument("--ablation", choices=tuple(ABLATIONS))
    mode.add_argument("--candidate-matrix", action="store_true",
                      help="Compare LRU, fixed, short_only and guarded across the interleaved budget matrix")
    parser.add_argument("--pinned", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--tracks", nargs="+", choices=tuple(BUDGETS))
    parser.add_argument("--budget", type=int, help="Use one cache budget in MiB")
    parser.add_argument("--blocks", type=Path, default=Path("data/playroom/blocks-2048-banded"))
    parser.add_argument("--workloads", type=Path, default=Path("reports/workloads/playroom"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 4 or args.rounds % 4:
        parser.error("Use a positive multiple of four for balanced order")
    if args.budget is not None and args.budget <= 0:
        parser.error("Cache budget must be positive")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True,
                          encoding="oem", check=True).stdout.strip()
    guid = re.search(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}", plan)
    session = {"power_plan": plan, "power_plan_guid": guid.group() if guid else None,
               "torch_threads": torch.get_num_threads(),
               "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
               "sampler_interval_s": 0.5, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0), "diagnostic": args.diagnose,
               "candidate_matrix": args.candidate_matrix}
    (args.output / "environment.json").write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
    print(session, flush=True)
    with torch.inference_mode(), Sampler() as sampler:
        try:
            tracks = args.tracks or (["return"] if args.diagnose else ["continuous"] if args.ablation else BUDGETS)
            for track in tracks:
                run_track(track, args, args.output / track, sampler)
                torch.cuda.empty_cache()
        finally:
            with (args.output / "telemetry.jsonl").open("w", encoding="utf-8") as stream:
                for sample in sampler.samples:
                    stream.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
