"""Trace H2D, D2D and render kernel execution with bundled CUPTI 12.4."""

import ctypes as c
import json
from pathlib import Path

import torch


MEMCPY, DRIVER, RUNTIME, CONCURRENT_KERNEL = 1, 4, 5, 10
HTOD, DTOD = 1, 8
ACTIVITIES = (MEMCPY, DRIVER, RUNTIME, CONCURRENT_KERNEL)
U32, U64, PTR, SIZE = c.c_uint32, c.c_uint64, c.c_void_p, c.c_size_t
REQUEST = c.CFUNCTYPE(None, c.POINTER(PTR), c.POINTER(SIZE), c.POINTER(SIZE))
COMPLETE = c.CFUNCTYPE(None, PTR, U32, PTR, SIZE, SIZE)


class CopyRecord(c.Structure):
    _fields_ = [("kind", U32), ("copy_kind", c.c_uint8), ("flags", c.c_uint8 * 3),
                ("bytes", U64), ("start", U64), ("end", U64), ("device", U32),
                ("context", U32), ("stream", U32), ("correlation", U32)]


class KernelRecord(c.Structure):
    _fields_ = [("kind", U32), ("config", U32 * 3), ("start", U64), ("end", U64),
                ("completed", U64), ("device", U32), ("context", U32), ("stream", U32),
                ("grid_block_and_memory", U32 * 10), ("correlation", U32)]


class ApiRecord(c.Structure):
    _fields_ = [("kind", U32), ("cbid", U32), ("start", U64), ("end", U64),
                ("process", U32), ("thread", U32), ("correlation", U32)]


def intersections(left, right):
    """Duration of the intersection of two internally disjoint interval lists."""
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return total


class ActivityTrace:
    def __init__(self, copy_stream):
        dll = Path(torch.__file__).parent / "lib" / "cupti64_2024.1.0.dll"
        self.lib = c.CDLL(str(dll))
        self.buffers, self.errors, self.frames = {}, [], []
        self.copies, self.kernels, self.copy_kernels, self.apis = [], [], [], []
        signatures = {
            "cuptiGetVersion": [c.POINTER(U32)],
            "cuptiGetResultString": [c.c_int, c.POINTER(c.c_char_p)],
            "cuptiGetTimestamp": [c.POINTER(U64)],
            "cuptiGetStreamIdEx": [PTR, PTR, c.c_uint8, c.POINTER(U32)],
            "cuptiActivityRegisterCallbacks": [REQUEST, COMPLETE],
            "cuptiActivityEnable": [U32], "cuptiActivityDisable": [U32],
            "cuptiActivityGetNextRecord": [PTR, SIZE, c.POINTER(PTR)],
            "cuptiActivityGetNumDroppedRecords": [PTR, U32, c.POINTER(SIZE)],
            "cuptiActivityFlushAll": [U32],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = arguments
            function.restype = c.c_int
        version = U32()
        self.check(self.lib.cuptiGetVersion(c.byref(version)))
        if version.value != 22:
            raise RuntimeError(f"Unsupported CUPTI record ABI: {version.value}")
        driver = c.WinDLL("nvcuda.dll")
        driver.cuCtxGetCurrent.argtypes = [c.POINTER(PTR)]
        context = PTR()
        if driver.cuCtxGetCurrent(c.byref(context)) or not context.value:
            raise RuntimeError("CUPTI requires an initialized CUDA context")
        self.copy_stream = self.stream_id(context, copy_stream.cuda_stream)
        self.render_stream = self.stream_id(context, torch.cuda.current_stream().cuda_stream)
        if self.copy_stream == self.render_stream:
            raise RuntimeError("Expected separate copy and render streams")
        self.request = REQUEST(self.request_buffer)
        self.complete = COMPLETE(self.complete_buffer)

    def check(self, code):
        if code:
            message = c.c_char_p()
            self.lib.cuptiGetResultString(code, c.byref(message))
            raise RuntimeError(f"CUPTI {code}: {message.value}")

    def stream_id(self, context, stream):
        result = U32()
        self.check(self.lib.cuptiGetStreamIdEx(context, stream, 0, c.byref(result)))
        return result.value

    def request_buffer(self, pointer, size, max_records):
        buffer = c.create_string_buffer(2**20)
        address = c.addressof(buffer)
        self.buffers[address] = buffer
        pointer[0], size[0], max_records[0] = address, len(buffer), 0

    def complete_buffer(self, context, stream, buffer, size, valid_size):
        try:
            cursor = PTR()
            while valid_size:
                code = self.lib.cuptiActivityGetNextRecord(buffer, valid_size, c.byref(cursor))
                if code == 12:  # CUPTI_ERROR_MAX_LIMIT_REACHED
                    break
                self.check(code)
                kind = U32.from_address(cursor.value).value
                if kind == MEMCPY:
                    record = CopyRecord.from_address(cursor.value)
                    if record.copy_kind in (HTOD, DTOD) and record.stream == self.copy_stream:
                        self.copies.append((record.start, record.end, record.bytes, record.copy_kind))
                elif kind == CONCURRENT_KERNEL:
                    record = KernelRecord.from_address(cursor.value)
                    if record.stream == self.render_stream:
                        self.kernels.append((record.start, record.end, record.correlation))
                    elif record.stream == self.copy_stream:
                        self.copy_kernels.append((record.start, record.end))
                elif kind in (DRIVER, RUNTIME):
                    record = ApiRecord.from_address(cursor.value)
                    self.apis.append((record.start, record.correlation))
            dropped = SIZE()
            self.check(self.lib.cuptiActivityGetNumDroppedRecords(context, stream, c.byref(dropped)))
            if dropped.value:
                self.errors.append(f"{dropped.value} CUPTI records dropped")
        except Exception as exc:
            self.errors.append(str(exc))
        finally:
            self.buffers.pop(buffer, None)

    def stamp(self):
        result = U64()
        self.check(self.lib.cuptiGetTimestamp(c.byref(result)))
        return result.value

    def start_frame(self):
        self.frames.append([self.stamp()])

    def mark(self):
        self.frames[-1].append(self.stamp())

    start_render = end_render = end_frame = mark

    def __enter__(self):
        self.check(self.lib.cuptiActivityRegisterCallbacks(self.request, self.complete))
        for kind in ACTIVITIES:
            self.check(self.lib.cuptiActivityEnable(kind))
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.check(self.lib.cuptiActivityFlushAll(0))
        for kind in reversed(ACTIVITIES):
            self.check(self.lib.cuptiActivityDisable(kind))

    def write(self, output, replay):
        rows = []
        for frame, ((start, render_start, render_end, end), expected) in enumerate(zip(self.frames, replay)):
            correlations = {correlation for time, correlation in self.apis if render_start <= time <= render_end}
            kernels = sorted((a, b) for a, b, correlation in self.kernels
                             if correlation in correlations and start <= a < b <= end)
            copies = sorted((a, b, size) for a, b, size, kind in self.copies
                            if kind == HTOD and start <= a < b <= end)
            device_copies = sorted((a, b, size) for a, b, size, kind in self.copies
                                   if kind == DTOD and start <= a < b <= end)
            copy_kernels = sorted((a, b) for a, b in self.copy_kernels if start <= a < b <= end)
            size = sum(copy[2] for copy in copies)
            device_bytes = sum(copy[2] for copy in device_copies)
            expected_device = expected["prefetch_d2d_logical_bytes"]
            if size != expected["prefetch_bytes"] or not kernels:
                self.errors.append(f"Frame {frame}: traced {size}/{expected['prefetch_bytes']} B, {len(kernels)} kernels")
            if device_bytes > expected_device or (device_bytes != expected_device and not copy_kernels):
                self.errors.append(f"Frame {frame}: D2D trace {device_bytes}/{expected_device} B")
            rows.append({"frame": frame, "prefetch_bytes": size, "copy_count": len(copies),
                         "render_kernel_count": len(kernels),
                         "copy_ms": sum(b - a for a, b, _ in copies) / 1e6,
                         "kernel_copy_overlap_ms": intersections([(a, b) for a, b, _ in copies], kernels) / 1e6,
                         "d2d_bytes": device_bytes, "d2d_count": len(device_copies),
                         "d2d_ms": sum(b - a for a, b, _ in device_copies) / 1e6,
                         "kernel_d2d_overlap_ms": intersections(
                             [(a, b) for a, b, _ in device_copies], kernels) / 1e6,
                         "copy_stream_kernel_count": len(copy_kernels),
                         "copy_stream_kernel_ms": sum(b - a for a, b in copy_kernels) / 1e6,
                         "event_window_overlap_ms": expected["prefetch_render_window_overlap_ms"]})
        report = {"source": "CUPTI activity timestamps", "errors": self.errors,
                  "copy_stream": self.copy_stream, "render_stream": self.render_stream,
                  "frames_with_copy": sum(row["copy_count"] > 0 for row in rows),
                  "frames_with_kernel_copy_overlap": sum(row["kernel_copy_overlap_ms"] > 0 for row in rows),
                  "frames": rows}
        (output / "device_trace.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if self.errors:
            raise RuntimeError(f"CUPTI trace verification failed: {self.errors[:3]}")
        print("Verified CUPTI overlap", report["frames_with_kernel_copy_overlap"], "/", report["frames_with_copy"])
