"""Compare short_only RGB to same-layout LRU without collecting timings."""

import argparse
import json
import math
from pathlib import Path

import torch

from .compare_prefetch import AsyncUpload, replay_once
from .run_reliable_benchmark import ABLATIONS, BUDGETS, SHORT_PIPELINE_TRAFFIC
from .stream_blocks import BlockCache, MIB, Replay, write_report


def audit(replay, budget, uploader, quota, output, vectorized=True, check_cache_order=False):
    replay.cache.budget = budget * MIB
    baseline = BlockCache(replay.blocks, replay.cache.budget)
    frames = []
    cache_order = []

    def compare(frame, rgb):
        required = replay.select(frame)
        baseline.prepare(required)
        inputs = baseline.assemble(required)
        reference, _, _ = replay.render(inputs, frame)
        error = (rgb - reference).abs().amax().item()
        if not math.isfinite(error):
            raise RuntimeError(f"Non-finite RGB difference at frame {frame}")
        frames.append({"frame": frame, "camera": replay.names[frame], "max_absolute_rgb_error": error})
        if check_cache_order:
            cache_order.append(tuple(replay.cache.entries))

    rows = replay_once(replay, "adaptive", uploader, quota, len(replay.names), observe=compare,
                       launch_order="render-first", vectorized=vectorized, **ABLATIONS["candidate"]["short_only"])
    worst = max(frames, key=lambda row: row["max_absolute_rgb_error"])
    error = worst["max_absolute_rgb_error"]
    report = {"status": "passed" if error == 0 else "rgb_mismatch",
              "policy": "short_only", "reference": "same_layout_lru", "budget_mib": budget,
              "batch": uploader.batch, "vectorized": vectorized,
              "frames": len(frames), "max_absolute_rgb_error": error,
              "identical_rgb_frames": sum(row["max_absolute_rgb_error"] == 0 for row in frames),
              "worst_frame": worst["frame"] if error else None}
    write_report(output, report, frames)
    if error:
        raise RuntimeError(f"RGB mismatch at {output} frame {worst['frame']}: {error}")
    if check_cache_order:
        return cache_order, [tuple(row[field] for field in SHORT_PIPELINE_TRAFFIC) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=Path, default=Path("data/playroom/blocks-2048-banded"))
    parser.add_argument("--workloads", type=Path, default=Path("reports/workloads/playroom"))
    parser.add_argument("--tracks", nargs="+", choices=tuple(BUDGETS), default=tuple(BUDGETS))
    parser.add_argument("--budget", type=int, help="Use one cache budget in MiB")
    parser.add_argument("--short-pipeline", action="store_true", help="Verify RGB and cache state for all four pipeline arms")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.budget is not None and args.budget <= 0:
        parser.error("Cache budget must be positive")
    torch.set_num_threads(1)
    quota = 16 * MIB
    with torch.inference_mode():
        for track in args.tracks:
            budgets = (args.budget,) if args.budget is not None else BUDGETS[track]
            replay = Replay(args.blocks, args.workloads / f"{track}.json", budgets[0] * MIB)
            uploader = AsyncUpload(replay.blocks, quota, batch=True)
            single = AsyncUpload(replay.blocks, quota) if args.short_pipeline else None
            for budget in budgets:
                output = args.output / track / str(budget)
                if not args.short_pipeline:
                    audit(replay, budget, uploader, quota, output)
                    continue
                reference = None
                for arm, options in ABLATIONS["short_pipeline"].items():
                    signature = audit(replay, budget, uploader if options["batch"] else single, quota,
                                      output / arm, vectorized=options["vectorized"], check_cache_order=True)
                    if reference is None:
                        reference = signature
                    if signature != reference:
                        raise RuntimeError(f"Cache order or traffic mismatch in {track}/{budget}/{arm}")
                report = {"status": "passed", "frames_per_arm": len(replay.names),
                          "arms": list(ABLATIONS["short_pipeline"]),
                          "identical_cache_order_and_traffic": True, "max_absolute_rgb_error": 0.0}
                (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(f"Four pipeline arms have identical frame cache order and traffic: {track}/{budget}", flush=True)
            del replay, uploader, single
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
