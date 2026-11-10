"""Benchmark a full-resident Graphdeco scene on the fixed gsplat backend."""

import argparse
import csv
import importlib.metadata
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from .graphdeco_scene import load_cameras, load_gaussians
from .render_settings import FAR, NEAR, render_scene
from .smoke_render import save_png


def timing_stats(values):
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def baseline(args):
    scene = args.scene
    torch.cuda.init()
    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    arrays, sh_degree = load_gaussians(scene / "point_cloud.ply")
    views, intrinsics, names = load_cameras(scene / "cameras.json", args.width, args.height)
    payload_bytes = sum(array.nbytes for array in arrays.values())
    count = len(arrays["means"])
    decoded = time.perf_counter()
    if payload_bytes >= free_before:
        raise RuntimeError(f"Full scene payload {payload_bytes} B exceeds free GPU memory {free_before} B")
    inputs = {key: torch.from_numpy(value).to(device="cuda") for key, value in arrays.items()}
    view_gpu = torch.from_numpy(views).to(device="cuda")
    intrinsics_gpu = torch.from_numpy(intrinsics).to(device="cuda")
    del arrays
    torch.cuda.synchronize()
    uploaded = time.perf_counter()
    resident_bytes = torch.cuda.memory_allocated()
    print(f"Full upload: {count:,} Gaussians, {payload_bytes / 2**20:.1f} MiB", flush=True)

    def render(index):
        return render_scene(inputs, view_gpu[index:index + 1], intrinsics_gpu[index:index + 1],
                            args.width, args.height, sh_degree)

    image, alpha, meta = render(0)
    torch.cuda.synchronize()
    first_done = time.perf_counter()
    if not torch.isfinite(image).all().item() or alpha.max().item() <= 0:
        raise RuntimeError("First render is empty or non-finite")
    del image, alpha, meta
    startup_peak = torch.cuda.max_memory_allocated()
    startup_reserved_peak = torch.cuda.max_memory_reserved()
    print(f"First frame: {(first_done - started) * 1000:.1f} ms from load start", flush=True)

    for index in np.linspace(0, len(names) - 1, args.warmup, dtype=int):
        render(int(index))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    rows, repeats = [], []
    for repeat in range(1, args.repeats + 1):
        wall, gpu = [], []
        for index, name in enumerate(names):
            begin = time.perf_counter()
            event_start.record()
            image, alpha, meta = render(index)
            event_end.record()
            event_end.synchronize()
            frame_ms = (time.perf_counter() - begin) * 1000
            gpu_ms = event_start.elapsed_time(event_end)
            wall.append(frame_ms)
            gpu.append(gpu_ms)
            rows.append({"repeat": repeat, "frame": index, "camera": name,
                         "wall_ms": frame_ms, "gpu_ms": gpu_ms,
                         "projected_gaussians": meta["gaussian_ids"].numel(),
                         "tile_intersections": meta["isect_ids"].numel()})
            del image, alpha, meta
        repeats.append({"repeat": repeat, "wall_ms": timing_stats(wall), "gpu_ms": timing_stats(gpu)})
        print(f"Pass {repeat}: wall mean {np.mean(wall):.2f} ms, p95 {np.percentile(wall, 95):.2f} ms",
              flush=True)

    free_after, _ = torch.cuda.mem_get_info()
    allocated_peak = torch.cuda.max_memory_allocated()
    reserved_peak = torch.cuda.max_memory_reserved()
    wall_ms = [row["wall_ms"] for row in rows]
    report = {
        "status": "full_scene_render_passed", "policy": "full_resident",
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "scene": scene.name, "source": json.loads((scene / "source.json").read_text(encoding="utf-8")),
        "gaussians": count, "sh_degree": sh_degree, "dtype": "float32",
        "resolution": [args.width, args.height], "packed": True, "radius_clip": 0.0,
        "near_plane": NEAR, "far_plane": FAR, "background": [0, 0, 0],
        "rasterize_mode": "classic", "camera_count": len(names),
        "warmup_frames": args.warmup, "repeats": args.repeats, "measured_frames": len(rows),
        "torch": torch.__version__, "gsplat": importlib.metadata.version("gsplat"),
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "startup_ms": {"ply_and_camera_decode": (decoded - started) * 1000,
                       "upload_and_camera_setup": (uploaded - decoded) * 1000,
                       "first_render": (first_done - uploaded) * 1000,
                       "load_to_first_frame": (first_done - started) * 1000},
        "memory_bytes": {"scene_payload": payload_bytes, "allocated_after_upload": resident_bytes,
                         "gpu_total": total, "gpu_free_before_load": free_before,
                         "gpu_free_after_replay": free_after,
                         "startup_peak_allocated": startup_peak, "startup_peak_reserved": startup_reserved_peak,
                         "steady_peak_allocated": allocated_peak, "steady_peak_reserved": reserved_peak},
        "wall_ms": timing_stats(wall_ms), "gpu_ms": timing_stats([row["gpu_ms"] for row in rows]),
        "render_throughput_fps": 1000 / float(np.mean(wall_ms)),
        "frames_over_33_3ms_fraction": sum(value > 33.3 for value in wall_ms) / len(rows),
        "passes": repeats,
        "timing_scope": "Synchronous offscreen frame completion. Image export runs after timing. "
                        "Startup excludes Python imports and CUDA context initialization.",
    }
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    selected = np.linspace(0, len(names) - 1, 4, dtype=int)
    previews = []
    for index in selected:
        image, alpha, _ = render(int(index))
        if not torch.isfinite(image).all().item() or alpha.max().item() <= 0:
            raise RuntimeError(f"Invalid preview at camera {names[index]}")
        pixels = (image[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        save_png(folder / f"frame_{index:04d}.png", args.width, args.height, pixels.tobytes())
        previews.append(pixels[::2, ::2])
    sheet = np.concatenate([np.concatenate(previews[:2], axis=1),
                            np.concatenate(previews[2:], axis=1)], axis=0)
    save_png(folder / "preview.png", sheet.shape[1], sheet.shape[0], sheet.tobytes())
    report["preview_camera_names"] = [names[index] for index in selected]
    with (folder / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / "trajectory.json").write_text(json.dumps(
        {"camera_names": names, "viewmats": views.tolist(), "intrinsics": intrinsics.tolist(),
         "resolution": [args.width, args.height]}, indent=2) + "\n", encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} measured frames to {folder.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.width, args.height, args.repeats) < 1 or args.warmup < 0:
        parser.error("Resolution/repeats must be positive and warmup nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    with torch.inference_mode():
        baseline(args)


if __name__ == "__main__":
    main()
