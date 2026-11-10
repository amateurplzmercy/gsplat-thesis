"""Read-only local checks. Uses only the standard library unless --probe-torch."""

import argparse
import ctypes
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


def command(args, timeout=20):
    try:
        result = subprocess.run(args, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def windows_memory():
    if os.name != "nt":
        return {"status": "not_checked_on_this_platform"}

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (key, ctypes.c_ulonglong) for key in
            ("physical_total", "physical_available", "page_total", "page_available",
             "virtual_total", "virtual_available", "extended")]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"status": "unavailable"}
    return {"physical_total_bytes": status.physical_total,
            "physical_available_bytes": status.physical_available}


def inspect(probe_torch=False):
    root = Path(__file__).resolve().parents[1]
    report = {
        "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "platform": platform.platform(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "project": str(root),
        "packages_discoverable": {name: importlib.util.find_spec(name) is not None
                                  for name in ("torch", "numpy", "gsplat")},
        "commands_on_path": {name: shutil.which(name) for name in
                             ("git", "nvidia-smi", "nvcc", "cl", "conda", "uv", "wsl")},
        "memory": windows_memory(),
        "disk": {},
        "notes": ["nvidia-smi CUDA version is driver capability, not Toolkit installation.",
                  "Missing PATH entries do not prove software is absent.",
                  "Free memory and disk space are point-in-time observations."],
    }
    for anchor in sorted({root.anchor, Path(sys.executable).anchor, "C:\\"}
                         if os.name == "nt" else {"/"}):
        try:
            usage = shutil.disk_usage(anchor)
            report["disk"][anchor] = {"total_bytes": usage.total, "free_bytes": usage.free}
        except OSError as exc:
            report["disk"][anchor] = {"error": str(exc)}
    if shutil.which("nvidia-smi"):
        report["nvidia_smi"] = command([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,nounits"])
    report["torch_probe"] = {"status": "not_requested"}
    if probe_torch:
        # Isolate import errors, GPU initialization and possible hangs in a child process.
        report["torch_probe"] = command([sys.executable, "-X", "utf8", "-s", "-c",
            "import json,torch; ok=torch.cuda.is_available(); "
            "print(json.dumps({'torch':torch.__version__, 'cuda_available':ok, "
            "'runtime':torch.version.cuda, "
            "'device':torch.cuda.get_device_name(0) if ok else None}))"], timeout=90)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-torch", action="store_true")
    args = parser.parse_args()
    report = inspect(args.probe_torch)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    # ASCII escaping keeps redirected Windows terminals usable without codepage changes.
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
