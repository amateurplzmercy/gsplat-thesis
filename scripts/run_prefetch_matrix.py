"""Run interleaved Playroom policy and budget comparisons on the optimized pipeline."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .compare_prefetch import AsyncUpload, replay_once
from .stream_blocks import MIB, Replay, write_report


ORDERS = (
    ("lru", "fixed", "guarded", "adaptive"),
    ("fixed", "adaptive", "lru", "guarded"),
    ("adaptive", "guarded", "fixed", "lru"),
    ("guarded", "lru", "adaptive", "fixed"),
)
CASES = {
    "main": (("continuous", 298), ("return", 298), ("capture_order", 375)),
    "budget": (("continuous", 326), ("return", 326), ("continuous", 430), ("return", 430)),
}


def summarize(rows):
    warm = rows[1:]
    totals = {key: sum(row[key] for row in warm) for key in (
        "upload_bytes", "prefetch_bytes", "prefetch_used_bytes", "prefetch_wasted_bytes",
        "evicted_blocks", "evicted_bytes", "prefetch_evicted_blocks", "prefetch_evicted_bytes",
        "prefetch_d2d_logical_bytes")}
    if totals["prefetch_bytes"] != totals["prefetch_used_bytes"] + totals["prefetch_wasted_bytes"]:
        raise RuntimeError("Prefetch traffic does not balance")
    total_h2d = totals["upload_bytes"] + totals["prefetch_bytes"]
    return {
        "p95_ms": float(np.percentile([row["wall_ms"] for row in warm], 95)),
        "p99_ms": float(np.percentile([row["wall_ms"] for row in warm], 99)),
        "warm_total_h2d_bytes": total_h2d,
        "cold_h2d_bytes": rows[0]["upload_bytes"] + rows[0]["prefetch_bytes"],
        "full_replay_h2d_bytes": total_h2d + rows[0]["upload_bytes"] + rows[0]["prefetch_bytes"],
        "warm_miss_frames": sum(row["upload_bytes"] > 0 for row in warm),
        "total_evicted_bytes": totals["evicted_bytes"] + totals["prefetch_evicted_bytes"],
        **totals,
    }


def measure(track, budget, blocks, workloads, output, repeats):
    replay = Replay(blocks, workloads / f"{track}.json", budget * MIB)
    feasibility = replay.feasibility()
    if feasibility["over_budget_frames"]:
        raise ValueError(f"{track} at {budget} MiB cannot fit all frames")
    quota = 16 * MIB
    uploader = AsyncUpload(replay.blocks, quota, batch=True)

    def run(policy, frames):
        return replay_once(replay, policy, None if policy == "lru" else uploader, quota, frames,
                           launch_order="render-first", vectorized=True)

    for policy in ORDERS[0]:
        run(policy, min(len(replay.names), 60))
    records, rounds = [], []
    reference = None
    for repeat in range(repeats):
        order = ORDERS[repeat % len(ORDERS)]
        trial = {"round": repeat + 1, "order": order, "policies": {}}
        for position, policy in enumerate(order, 1):
            rows = run(policy, len(replay.names))
            signature = [(row["required_bytes"], row["projected_gaussians"]) for row in rows]
            if reference is None:
                reference = signature
            if signature != reference or any(row["resident_bytes"] > budget * MIB for row in rows):
                raise RuntimeError("Frame coverage or cache budget changed across policy runs")
            trial["policies"][policy] = summarize(rows)
            for row in rows:
                records.append({"round": repeat + 1, "policy": policy, "order_position": position, **row})
        rounds.append(trial)
        times = ", ".join(f"{policy} {result['p99_ms']:.2f}"
                          for policy, result in trial["policies"].items())
        print(f"{track} {budget} MiB, round {repeat+1}/{repeats}, p99 ms: {times}", flush=True)
    medians = {}
    for policy in ORDERS[0]:
        runs = [trial["policies"][policy] for trial in rounds]
        medians[policy] = {key: float(np.median([result[key] for result in runs])) for key in runs[0]}
    comparisons = {}
    for policy in ("fixed", "adaptive", "guarded"):
        for baseline in ("lru", "fixed"):
            if baseline == policy:
                continue
            diffs = [trial["policies"][policy]["p99_ms"] - trial["policies"][baseline]["p99_ms"]
                     for trial in rounds]
            comparisons[f"{policy}_minus_{baseline}"] = {
                "p99_delta_ms_by_round": diffs, "median_p99_delta_ms": float(np.median(diffs)),
                "rounds_lower": sum(delta < 0 for delta in diffs),
            }
    report = {"status": "passed", "track": track, "budget_mib": budget,
              "frames_per_replay": len(replay.names), "repeats": repeats, "quota_mib": 16,
              "pipeline": ["render-first", "batch-upload", "vectorized"],
              "warm_scope": "All frames after frame zero; cache resets each replay",
              "medians": medians, "paired": comparisons, "rounds": rounds}
    write_report(output / f"{track}-{budget}", report, records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(CASES), required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--blocks", type=Path, default=Path("data/playroom/blocks-2048-banded"))
    parser.add_argument("--workloads", type=Path, default=Path("reports/workloads/playroom"))
    parser.add_argument("--output", type=Path, default=Path("reports/prefetch-study"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Repeats must be positive")
    output = args.output / args.phase
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        summaries = []
        for track, budget in CASES[args.phase]:
            summaries.append(measure(track, budget, args.blocks, args.workloads, output, args.repeats))
            torch.cuda.empty_cache()
    (output / "matrix.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
