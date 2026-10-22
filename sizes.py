"""Generation sizes from the final output size.

The high-resolution pass runs at 1/2 of the output, but never with a short side below min_short_side
(512 = the validated 512x896); the low-resolution pass is half of the high pass. Both are on the H3
32 px grid. ESRGAN x4 followed by esrgan_scale_by covers the output size and the save node center-crops
to it, so the output is exactly the requested size without stretching (a few % of the edge is cut when
the grid changes the aspect). Small faces (wide shots, full body) lose quality at these sizes.
"""
GRID = 32


def _snap(value):
    return max(GRID, round(value / GRID) * GRID)


class FastH3AccelSizes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "output_width": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 32}),
            "output_height": ("INT", {"default": 1792, "min": 128, "max": 8192, "step": 32}),
            "min_short_side": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64,
                                       "tooltip": "Smallest short side of the generation size (high pass)."}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "low_width", "low_height", "esrgan_scale_by", "crop_width", "crop_height")
    FUNCTION = "sizes"
    CATEGORY = "MiniMax H3/FastH3 Accel"
    DESCRIPTION = ("Generation size (1/2 of the output, short side at least min_short_side), first-pass size "
                   "(half of that) and the resize after the x4 ESRGAN model.")

    def sizes(self, output_width, output_height, min_short_side):
        factor = max(0.5, min_short_side / min(output_width, output_height))
        low_w, low_h = _snap(output_width * factor / 2), _snap(output_height * factor / 2)
        width, height = low_w * 2, low_h * 2
        scale_by = max(output_width / (width * 4), output_height / (height * 4))
        summary = (f"{low_w}x{low_h} -> {width}x{height} -> {round(width * 4 * scale_by)}x{round(height * 4 * scale_by)}"
                   f" -> crop {output_width}x{output_height}")
        return {"ui": {"text": [summary]}, "result": (width, height, low_w, low_h, scale_by, output_width, output_height)}


NODE_CLASS_MAPPINGS = {"FastH3AccelSizes": FastH3AccelSizes}
NODE_DISPLAY_NAME_MAPPINGS = {"FastH3AccelSizes": "FastH3 Accel Sizes (from output size)"}
