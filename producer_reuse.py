"""Reuse the first-pass QKV / VSA-gate projections in ComfyUI's native H3 sparse producer.

On the first step of every sampler call (no pooled kmean/vscale statistics yet) the native
h3_sparse_attention streams its token chunks through comfy-kitchen twice: once to collect the
statistics and once for the attention itself. Both passes project the same rows through qkv_proj
and to_gate_compress. This node keeps the first-pass outputs (up to a VRAM budget, QKV first) and
returns them on the second pass, so the result is bit-identical with less work.

Place it directly after Block Sparse Attention (method vsa or sla). Requires the native block
patches installed by that node.
"""
import inspect
import math
import sys


PRODUCER_CHUNK = 4096
MIB = 1024 * 1024


class _ReuseProjection:
    def __init__(self, projection, chunk_count, store, budget):
        self.projection, self.chunk_count, self.store, self.budget = projection, chunk_count, store, budget
        self.calls = 0
        self.cache = {}

    def __call__(self, x):
        index = self.calls
        self.calls += 1
        chunk, second = index % self.chunk_count, index >= self.chunk_count
        cached = self.cache.pop(chunk, None) if second else None
        if cached is not None and cached[0] == (tuple(x.shape), x.dtype, x.device):
            return cached[1]
        out = self.projection(x)
        if self.store and not second:
            size = out.numel() * out.element_size()
            if size <= self.budget[0]:
                self.budget[0] -= size
                self.cache[chunk] = ((tuple(x.shape), x.dtype, x.device), out)
        return out


class _AttentionProxy:
    def __init__(self, attention, qkv, gate):
        self._attention = attention
        self.qkv_proj = qkv
        self.to_gate_compress = gate

    def __getattr__(self, name):
        return getattr(self._attention, name)


def _native(function):
    module = sys.modules.get(getattr(function, "__module__", None))
    if module is None or not hasattr(module, "h3_sparse_attention") or module.__dict__ is not function.__globals__:
        raise ValueError("Place this node directly after native Block Sparse Attention.")
    return module


def _make_block_patch(native, block, block_index, patch, budget_bytes):
    def attention(h, rope_freqs=None, transformer_options={}):
        n = h.shape[0]
        if patch.vsa:
            n = patch.vsa_plan(transformer_options["minimax_h3_layout"], h.device)["n"]
        key = (block_index, n, tuple(transformer_options.get("uuids", ())))
        first = patch.pooled.get(key) is None
        chunk_count = math.ceil(n / PRODUCER_CHUNK)
        budget = [budget_bytes]
        qkv = _ReuseProjection(block.attn.qkv_proj, chunk_count, first, budget)
        gate = block.attn.to_gate_compress
        if gate is not None:
            gate = _ReuseProjection(gate, chunk_count, first, budget)
        try:
            return native.h3_sparse_attention(_AttentionProxy(block.attn, qkv, gate), h, rope_freqs,
                                              transformer_options, patch, block_index)
        finally:
            qkv.cache.clear()
            if gate is not None:
                gate.cache.clear()

    def block_patch(args, extra):
        if native.h3_eligible(block.attn, args["img"], args["rope_freqs"], args["transformer_options"], patch, block_index):
            args = {**args, "attention": attention}
        return extra["original_block"](args)

    return block_patch


class FastH3AccelProducerReuse:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "cache_mib": ("INT", {"default": 1280, "min": 0, "max": 8192, "step": 64,
                                  "tooltip": "VRAM for first-pass projections per attention call (QKV chunks first, then gate)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/FastH3 Accel"

    def patch(self, model, cache_mib):
        blocks = model.get_model_object("diffusion_model.blocks")
        replacements = model.model_options["transformer_options"].get("patches_replace", {}).get("dit", {})
        patched = model.clone()
        for index, block in enumerate(blocks):
            existing = replacements.get(("double_block", index))
            if existing is None:
                raise ValueError("Place this node directly after native Block Sparse Attention.")
            native = _native(existing)
            captured = inspect.getclosurevars(existing).nonlocals
            patch = captured.get("patch")
            expected = native.make_h3_block_patch(block, index, patch)
            if existing.__code__ is not expected.__code__ or captured.get("block") is not block:
                raise ValueError("Block patch is not the native sparse producer (another wrapper is installed).")
            patched.set_model_patch_replace(_make_block_patch(native, block, index, patch, cache_mib * MIB),
                                            "dit", "double_block", index)
        return (patched,)


NODE_CLASS_MAPPINGS = {"FastH3AccelProducerReuse": FastH3AccelProducerReuse}
NODE_DISPLAY_NAME_MAPPINGS = {"FastH3AccelProducerReuse": "FastH3 Accel Sparse Producer Reuse"}
