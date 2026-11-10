"""Compare two actual block layouts under LRU with balanced replay order."""

import argparse
import csv
import json
from pathlib import Path

import torch

from .compare_prefetch import replay_once
from .run_prefetch_matrix import summarize
from .run_reliable_benchmark import allocator_counts
from .stream_blocks import MIB, Replay
from .system_telemetry import Sampler, payload_residency, pretouch, process_memory


def compare(args, sampler):
    replays = {name: Replay(path, args.trajectory, args.budget * MIB)
               for name, path in (("coarse", args.coarse), ("fine", args.fine))}
    if replays["coarse"].blocks.index["source"] != replays["fine"].blocks.index["source"]:
        raise ValueError("The two layouts must come from the same model")
    for replay in replays.values():
        if replay.feasibility()["over_budget_frames"]:
            raise ValueError("A block layout cannot fit this cache budget")
        replay_once(replay, "lru", None, 16 * MIB, min(60, len(replay.names)), launch_order="render-first")
        replay.cache.clear()
    track = args.trajectory.stem
    output = args.output / track
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    frame_count = len(replays["coarse"].names)
    projections = None
    with (output / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = None
        for round_number in range(1, args.rounds + 1):
            order = tuple(replays) if round_number % 2 else tuple(reversed(replays))
            for position, name in enumerate(order, 1):
                replay = replays[name]
                pretouch(replay.blocks.payload)
                label = f"{track}/{round_number}/{args.budget}/{name}"
                resident_before = payload_residency(replay.blocks.payload)
                before = sampler.sample(label + "/before")
                allocations = allocator_counts()
                rows = replay_once(replay, "lru", None, 16 * MIB, frame_count, launch_order="render-first",
                                   fault_counter=lambda: process_memory()[0])
                after = sampler.sample(label + "/after")
                final_allocations = allocator_counts()
                resident_after = payload_residency(replay.blocks.payload)
                replay.cache.clear()
                current = [row["projected_gaussians"] for row in rows]
                if projections is None:
                    projections = current
                if current != projections or any(row["resident_bytes"] > args.budget * MIB for row in rows):
                    raise RuntimeError(f"Projection count or budget mismatch in {label}")
                tags = {"round": round_number, "budget_mib": args.budget, "policy": name,
                        "position": position, "budget_position": 1}
                runs.append({**tags, **summarize(rows), "before": before, "after": after,
                             "payload_before": resident_before, "payload_after": resident_after,
                             "allocator_deltas": {key: final_allocations[key] - value if value is not None else None
                                                  for key, value in allocations.items()}})
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=[*tags, *rows[0]])
                    writer.writeheader()
                writer.writerows({**tags, **row} for row in rows)
            stream.flush()
            print(f"{track} layout pair {round_number}/{args.rounds}", flush=True)
    report = {"track": track, "rounds": args.rounds, "order_size": 2,
              "budgets_mib": [args.budget], "diagnostic": False,
              "ablation": "layout", "ablation_contrasts": [["fine", "coarse"]],
              "frame_count": frame_count, "layouts": {name: str(path) for name, path in
                                                     (("coarse", args.coarse), ("fine", args.fine))},
              "runs": runs}
    (output / "runs.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=375)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or args.budget <= 0:
        parser.error("Use an even number of rounds and a positive cache budget")
    torch.set_num_threads(1)
    with torch.inference_mode(), Sampler() as sampler:
        compare(args, sampler)
    with (args.output / "telemetry.jsonl").open("w", encoding="utf-8") as stream:
        for sample in sampler.samples:
            stream.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
