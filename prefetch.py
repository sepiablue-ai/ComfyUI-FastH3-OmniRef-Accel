"""Fault in a diffusion model's memory-mapped weights on a background thread.

With Dynamic VRAM the checkpoint stays memory-mapped and the first sampler step pays the page
faults for every weight (about 6 s for the 21 GB H3 DiT on an RTX 4070 machine). This node starts
a thread that touches one byte per page of every CPU weight, so that work overlaps whatever runs
next (the text encoder). CLIP is passed through only to order this node before text encoding.
Weights are read, never modified.
"""
import threading
import time
import logging

import torch

PAGE = 4096


def _cpu_storages(model):
    seen = set()
    for tensor in [*model.parameters(), *model.buffers()]:
        parts = [tensor]
        qdata = getattr(tensor, "_qdata", None)
        if qdata is not None:
            parts = [qdata, getattr(getattr(tensor, "_params", None), "scale", None)]
        for part in parts:
            if isinstance(part, torch.Tensor) and part.device.type == "cpu" and part.numel():
                key = part.untyped_storage().data_ptr()
                if key not in seen:
                    seen.add(key)
                    yield part


def _touch(tensors, label):
    begin = time.perf_counter()
    total = 0
    with torch.inference_mode():
        for tensor in tensors:
            raw = torch.empty(0, dtype=torch.uint8).set_(tensor.untyped_storage())
            if raw.numel():
                raw[::PAGE].sum()  # one read per page
                total += raw.numel()
    logging.info("[FastH3 Accel prefetch] %s: %.1f GB faulted in %.2fs", label, total / 1e9, time.perf_counter() - begin)


class FastH3AccelPrefetch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",)}}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "prefetch"
    CATEGORY = "MiniMax H3/FastH3 Accel"

    def prefetch(self, model, clip):
        tensors = list(_cpu_storages(model.model))
        threading.Thread(target=_touch, args=(tensors, type(model.model).__name__), daemon=True,
                         name="fasth3-accel-prefetch").start()
        return (model, clip)


NODE_CLASS_MAPPINGS = {"FastH3AccelPrefetch": FastH3AccelPrefetch}
NODE_DISPLAY_NAME_MAPPINGS = {"FastH3AccelPrefetch": "FastH3 Accel Prefetch DiT Weights"}
