"""Timestamped Windows process counters and read-only NVIDIA telemetry."""

import ctypes as c
import mmap
import os
import threading
import time

U32, U64, PTR, SIZE = c.c_uint32, c.c_uint64, c.c_void_p, c.c_size_t
kernel = c.WinDLL("kernel32.dll")
psapi = c.WinDLL("psapi.dll")
kernel.GetCurrentProcess.restype = PTR
process = kernel.GetCurrentProcess()


class ProcessMemory(c.Structure):
    _fields_ = [("size", U32), ("faults", U32), ("peak_working_set", SIZE),
                ("working_set", SIZE)] + [(f"unused_{i}", SIZE) for i in range(6)]


class Page(c.Structure):
    _fields_ = [("address", PTR), ("flags", SIZE)]


class GpuMemory(c.Structure):
    _fields_ = [("total", U64), ("free", U64), ("used", U64)]


class GpuUtilization(c.Structure):
    _fields_ = [("gpu", U32), ("memory", U32)]


class PowerStatus(c.Structure):
    _fields_ = [("ac", c.c_ubyte), ("flags", c.c_ubyte), ("battery", c.c_ubyte),
                ("saver", c.c_ubyte), ("life", U32), ("full", U32)]


psapi.GetProcessMemoryInfo.argtypes = [PTR, c.POINTER(ProcessMemory), U32]
psapi.QueryWorkingSetEx.argtypes = [PTR, PTR, U32]
kernel.GetSystemTimes.argtypes = [c.POINTER(U64)] * 3
kernel.GetSystemPowerStatus.argtypes = [c.POINTER(PowerStatus)]


def process_memory():
    values = ProcessMemory()
    values.size = c.sizeof(values)
    if not psapi.GetProcessMemoryInfo(process, c.byref(values), values.size):
        raise c.WinError()
    return values.faults, values.working_set


def payload_residency(payload):
    addresses = range(payload.data_ptr(), payload.data_ptr() + payload.nbytes, 16 * mmap.PAGESIZE)
    pages = (Page * len(addresses))(*(Page(address, 0) for address in addresses))
    if not psapi.QueryWorkingSetEx(process, pages, c.sizeof(pages)):
        raise c.WinError()
    return {"sampled_pages": len(pages), "resident_pages": sum(page.flags & 1 for page in pages)}


def pretouch(payload):
    array = payload.numpy()
    return float(array[::mmap.PAGESIZE // array.itemsize].sum(dtype="float64")) + float(array[-1])


class Sampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.lib = c.WinDLL("nvml.dll")
        self.handle = PTR()
        self.samples = []
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.previous = None
        signatures = {
            "nvmlDeviceGetHandleByIndex_v2": [U32, c.POINTER(PTR)],
            "nvmlDeviceGetClockInfo": [PTR, U32, c.POINTER(U32)],
            "nvmlDeviceGetTemperature": [PTR, U32, c.POINTER(U32)],
            "nvmlDeviceGetPowerUsage": [PTR, c.POINTER(U32)],
            "nvmlDeviceGetUtilizationRates": [PTR, c.POINTER(GpuUtilization)],
            "nvmlDeviceGetMemoryInfo": [PTR, c.POINTER(GpuMemory)],
            "nvmlDeviceGetPerformanceState": [PTR, c.POINTER(U32)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, c.c_int

    def gpu_value(self, method, *args):
        value = U32()
        code = getattr(self.lib, method)(self.handle, *args, c.byref(value))
        return value.value if code == 0 else None

    def sample(self, label="periodic"):
        with self.lock:
            now, process_ns = time.perf_counter_ns(), time.process_time_ns()
            idle, kern, user = U64(), U64(), U64()
            if not kernel.GetSystemTimes(c.byref(idle), c.byref(kern), c.byref(user)):
                raise c.WinError()
            current = (now, process_ns, idle.value, kern.value + user.value)
            cpu = proc_cpu = None
            if self.previous:
                prev_time, prev_proc, prev_idle, prev_total = self.previous
                elapsed, total = now - prev_time, current[3] - prev_total
                if total > 0:
                    cpu = 100 * (1 - (idle.value - prev_idle) / total)
                if elapsed > 0:
                    proc_cpu = 100 * (process_ns - prev_proc) / (elapsed * (os.cpu_count() or 1))
            self.previous = current
            faults, working_set = process_memory()
            status = PowerStatus()
            ac_known = kernel.GetSystemPowerStatus(c.byref(status))
            utilization, memory = GpuUtilization(), GpuMemory()
            util_ok = not self.lib.nvmlDeviceGetUtilizationRates(self.handle, c.byref(utilization))
            mem_ok = not self.lib.nvmlDeviceGetMemoryInfo(self.handle, c.byref(memory))
            power = self.gpu_value("nvmlDeviceGetPowerUsage")
            row = {
                "perf_ns": now, "label": label, "cpu_percent": cpu, "process_cpu_percent": proc_cpu,
                "page_faults": faults, "working_set_bytes": working_set,
                "ac_online": status.ac if ac_known else None,
                "sm_mhz": self.gpu_value("nvmlDeviceGetClockInfo", 1),
                "mem_mhz": self.gpu_value("nvmlDeviceGetClockInfo", 2),
                "temperature_c": self.gpu_value("nvmlDeviceGetTemperature", 0),
                "power_w": power / 1000 if power is not None else None,
                "gpu_percent": utilization.gpu if util_ok else None,
                "gpu_memory_used_bytes": memory.used if mem_ok else None,
                "pstate": self.gpu_value("nvmlDeviceGetPerformanceState"),
            }
            self.samples.append(row)
            return row

    def loop(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except OSError as exc:
                with self.lock:
                    self.samples.append({"perf_ns": time.perf_counter_ns(), "error": str(exc)})

    def __enter__(self):
        if self.lib.nvmlInit_v2():
            raise RuntimeError("NVML initialization failed")
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0, c.byref(self.handle)):
            self.lib.nvmlShutdown()
            raise RuntimeError("NVML could not open GPU 0")
        self.sample("session-start")
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.lib.nvmlShutdown()
