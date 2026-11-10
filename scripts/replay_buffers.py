"""Bounded host staging and reusable GPU packing storage."""

import time
import numpy as np
import torch

from .spatial_blocks import FIELDS
from .stream_blocks import MIB


class PinnedDemand:
    def __init__(self, blocks, slot_mib=16, slots=4):
        self.blocks = blocks
        self.elements = slot_mib * MIB // 4
        if slots < 1 or int(blocks.sizes.max()) > 4 * self.elements:
            raise ValueError("Pinned slot must fit the largest block")
        self.staging = torch.empty((slots, self.elements), dtype=torch.float32, pin_memory=True)
        self.host = self.staging.numpy()
        self.events = [torch.cuda.Event() for _ in range(slots)]
        self.begin_frame()

    def begin_frame(self):
        self.slot = self.offset = self.filled_slots = 0
        self.stage_ms = self.reuse_wait_ms = 0.0

    def __call__(self, block):
        source = self.blocks.cpu_block(block)
        count = source.numel()
        if self.offset + count > self.elements:
            self.events[self.slot].record()
            self.filled_slots += 1
            self.slot = (self.slot + 1) % len(self.events)
            if self.filled_slots >= len(self.events):
                start = time.perf_counter()
                self.events[self.slot].synchronize()
                self.reuse_wait_ms += (time.perf_counter() - start) * 1000
            self.offset = 0
        span = slice(self.offset, self.offset + count)
        start = time.perf_counter()
        np.copyto(self.host[self.slot, span], source.numpy())
        self.stage_ms += (time.perf_counter() - start) * 1000
        flat = self.staging[self.slot, span].to("cuda", non_blocking=True)
        self.offset += count
        return self.blocks.fields(flat, self.blocks.blocks[block]["count"])


def packing_buffer(blocks, peak_bytes):
    count, remainder = divmod(peak_bytes, 4 * blocks.stride)
    if remainder:
        raise ValueError("Packing size must align with Gaussian payload")
    return {field: torch.empty((count, *blocks.shapes[field]), dtype=torch.float32, device="cuda")
            for field in FIELDS}
