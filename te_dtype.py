"""Run the MiniMax H3 Qwen3-VL text encoder LLM in BF16 instead of ComfyUI's fixed FP32 execution.

sd1_clip.SDClipModel.forward always calls the transformer with dtype=torch.float32 and FP32 input
embeddings, so every quantized LLM linear is dequantized and run as an FP32 SGEMM. Casting the
embeddings to BF16 lets the quantized kernels run. The vision tower stays FP32 (BF16 vision breaks
the conditioning). Outputs are still returned as FP32.

The patch is an object patch on a CLIP clone, so the input CLIP is left unchanged.
"""
import torch


class FastH3AccelTEBF16:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",)}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/FastH3 Accel"
    DESCRIPTION = "Compute the MiniMax H3 text encoder LLM in BF16 (ComfyUI uses FP32 for it)."

    def patch(self, clip):
        outer = clip.cond_stage_model
        name = outer.clip
        transformer = getattr(outer, name).transformer
        if not hasattr(transformer, "visual"):
            raise ValueError("Connect a MiniMax H3 (Qwen3-VL) text encoder.")
        forward = transformer.forward

        def bf16_forward(input_ids, attention_mask=None, embeds=None, dtype=None, **kwargs):
            if embeds is not None:
                embeds = embeds.to(torch.bfloat16)
            return forward(input_ids, attention_mask=attention_mask, embeds=embeds, dtype=torch.bfloat16, **kwargs)

        patched = clip.clone()
        patched.patcher.add_object_patch(f"{name}.transformer.forward", bf16_forward)
        return (patched,)


NODE_CLASS_MAPPINGS = {"FastH3AccelTEBF16": FastH3AccelTEBF16}
NODE_DISPLAY_NAME_MAPPINGS = {"FastH3AccelTEBF16": "FastH3 Accel Text Encoder BF16"}
