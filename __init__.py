"""Unofficial ComfyUI acceleration nodes for FastH3 OmniRef (MiniMax H3 8-step Ref2VA), RTX 4070 12 GB validated."""
from . import ffn, lbh, prefetch, producer_reuse, sizes, stream, te_dtype

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for module in (sizes, te_dtype, prefetch, ffn, producer_reuse, lbh, stream):
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
