"""Render locally generated Gaussians on CUDA. Not a pretrained-scene benchmark."""

import argparse
import importlib.metadata
import json
import struct
import sys
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path


def save_png(path, width, height, data):
    """Write RGB8 PNG without adding an imaging dependency."""
    def chunk(kind, content):
        return (struct.pack(">I", len(content)) + kind + content
                + struct.pack(">I", zlib.crc32(kind + content) & 0xffffffff))
    rows = b"".join(b"\x00" + data[y * width * 3:(y + 1) * width * 3]
                    for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/smoke"))
    args = parser.parse_args()
    try:
        import torch
        from gsplat import rasterization
    except ImportError as exc:
        print(f"Required environment is not ready: {exc}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("A working CUDA PyTorch installation and NVIDIA GPU are required.", file=sys.stderr)
        return 2

    # All points are in front of an identity camera looking along +Z.
    # RGB and identity quaternions avoid SH/convention issues in this first smoke check.
    device = "cuda"
    width, height = 320, 240
    with torch.inference_mode():
        axis = torch.linspace(-0.8, 0.8, 24, device=device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        means = torch.stack((xx.flatten(), yy.flatten(),
                             torch.full_like(xx.flatten(), 3.0)), dim=-1).contiguous()
        count = means.shape[0]
        quats = torch.zeros((count, 4), device=device)
        quats[:, 0] = 1.0
        scales = torch.full((count, 3), 0.045, device=device)
        opacities = torch.full((count,), 0.85, device=device)
        colors = torch.stack(((means[:, 0] + 0.8) / 1.6,
                              (means[:, 1] + 0.8) / 1.6,
                              torch.full((count,), 0.5, device=device)), dim=-1)
        viewmats = torch.eye(4, device=device)[None]
        intrinsics = torch.tensor([[[250.0, 0.0, width / 2],
                                    [0.0, 250.0, height / 2],
                                    [0.0, 0.0, 1.0]]], device=device)

        def render():
            return rasterization(means=means, quats=quats, scales=scales,
                                 opacities=opacities, colors=colors,
                                 viewmats=viewmats, Ks=intrinsics,
                                 width=width, height=height,
                                 sh_degree=None, packed=True, radius_clip=0.0)

        first_start = time.perf_counter()
        image, alpha, _ = render()
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - first_start) * 1000
        if image.shape != (1, height, width, 3):
            raise RuntimeError(f"Unexpected image shape: {tuple(image.shape)}")
        if not bool(torch.isfinite(image).all()) or not bool(torch.isfinite(alpha).all()):
            raise RuntimeError("Non-finite render output")
        if float(alpha.max().item()) <= 0.01:
            raise RuntimeError("Empty render; camera or backend conventions need inspection")

        # Warm only this tiny interface test; these timings are not thesis results.
        for _ in range(3):
            render()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        image, alpha, _ = render()
        end.record()
        end.synchronize()
        gpu_ms = start.elapsed_time(end)
        rgb = (image[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().contiguous()
        from gsplat.cuda._backend import _C
        report = {
            "status": "cuda_render_passed",
            "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "compiled_backend": str(_C.__file__),
            "test_type": "synthetic_interface_smoke_not_performance_benchmark",
            "torch": torch.__version__,
            "gsplat": importlib.metadata.version("gsplat"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "resolution": [width, height],
            "gaussians": count,
            "first_call_including_possible_initialization_ms": first_call_ms,
            "single_warm_call_gpu_ms_for_diagnostics_only": gpu_ms,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "alpha_max": float(alpha.max().item()),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ppm = args.output_dir / "synthetic.ppm"
        pixels = rgb.numpy().tobytes()
        ppm.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)
        save_png(args.output_dir / "synthetic.png", width, height, pixels)
        (args.output_dir / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
