"""Export same-layout Playroom RGB and measured replay statistics for offline viewing."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.functional import interpolate

from .compare_prefetch import replay_once
from .smoke_render import save_png
from .stream_blocks import MIB, Replay


TRACKS = {"continuous": 326, "capture_order": 375}
POLICIES = ("lru", "fixed", "short_only")
BYTE_FIELDS = {"demand": "upload_bytes", "prefetch": "prefetch_bytes", "resident": "resident_bytes"}


def read_measurements(folder, budget, frame_count):
    report = json.loads((folder / "paired_statistics.json").read_text(encoding="utf-8"))
    paired = defaultdict(list)
    with (folder / "frames.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["budget_mib"]) == budget and row["policy"] in POLICIES:
                paired[row["policy"], int(row["frame"])].append(row)
    result = {}
    for policy in POLICIES:
        frames = []
        for frame in range(frame_count):
            rows = paired[policy, frame]
            if {int(row["round"]) for row in rows} != set(range(1, report["rounds"] + 1)):
                raise ValueError(f"Missing repeats at {folder.name}/{policy}/{frame}")
            values = {key: {int(row[field]) for row in rows} for key, field in BYTE_FIELDS.items()}
            if any(len(group) != 1 for group in values.values()):
                raise ValueError(f"Frame traffic differs across repeats at {folder.name}/{policy}/{frame}")
            frames.append({"wall": round(statistics.median(float(row["wall_ms"]) for row in rows), 3),
                           **{key: round(group.pop() / MIB, 3) for key, group in values.items()}})
        group = report["groups"][f"{budget}/{policy}"]
        totals = group["warm_traffic_min_median_max_bytes"]
        result[policy] = {"frames": frames, "p99": group["latency"]["p99_ms"]["estimate"],
                          "transfer": totals["warm_total_h2d_bytes"][1] / MIB,
                          "wasted": totals["prefetch_wasted_bytes"][1] / MIB,
                          "slow": group["frames_over_33_3ms"], "samples": group["frames"]}
    return result


def export(output, measurements, workloads):
    result = {}
    for track, budget in TRACKS.items():
        replay = Replay(Path("data/playroom/blocks-2048-banded"),
                        workloads / f"{track}.json", budget * MIB)
        folder = output / "media" / track
        folder.mkdir(parents=True, exist_ok=True)

        def save_frame(frame, image):
            width, height = 640, 360
            rgb = interpolate(image[..., :3].permute(0, 3, 1, 2), size=(height, width), mode="area")
            rgb = (rgb[0].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            save_png(folder / f"{frame:04d}.png", width, height, rgb.tobytes())

        frame_count = len(replay.names)
        result[track] = {"budget": budget, "cameras": replay.names,
                         "policies": read_measurements(measurements / track, budget, frame_count)}
        replay_once(replay, "lru", None, 16 * MIB, frame_count, observe=save_frame, launch_order="render-first")
        print(f"{track}: {frame_count} RGB frames and measured metrics exported", flush=True)
        del replay
        torch.cuda.empty_cache()
    (output / "replay-data.js").write_text("const REPLAY_DATA = " + json.dumps(result, separators=(",", ":"))
                                          + ";\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo"))
    parser.add_argument("--measurements", type=Path, default=Path("reports/measurement-repair/candidate-matrix"))
    parser.add_argument("--workloads", type=Path, default=Path("reports/workloads/playroom"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        export(args.output, args.measurements, args.workloads)


if __name__ == "__main__":
    main()
