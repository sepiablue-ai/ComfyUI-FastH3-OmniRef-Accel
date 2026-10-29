"""H3 VAE decode -> model upscale + resize -> NVENC, streamed per finalized VAE chunk.

The native H3 VAE writes finalized temporal chunks into `output_buffer`; each chunk is upscaled on
the GPU right away and handed to one NVENC worker thread, so decode, upscale and encode overlap and
the full-resolution video never exists in memory. Encoder settings follow ComfyUI-H3-X2-Stream.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import tempfile

import av
import torch
import torch.nn.functional as F
import comfy.model_management as mm
import folder_paths


def color_srgb(target):
    # BT.709 primaries/matrix, IEC 61966-2-1 transfer, limited range (same as ComfyUI's sRGB writer)
    target.color_primaries = 1
    target.color_trc = 13
    target.colorspace = 1
    target.color_range = 1


class UpscaleSink:
    def __init__(self, output, stream, shape, model, scale_by, method, batch_frames, out_hw, crop):
        # decode_temporal reads shape[2] to clip the padded tail
        self.output, self.stream, self.shape, self.frames = output, stream, tuple(shape), shape[2]
        self.model, self.scale_by, self.method, self.batch_frames, self.out_hw = model, scale_by, method, batch_frames, out_hw
        self.crop = crop  # (top, left, height, width) center crop after the resize
        self.written = 0
        self.pending = None
        self.future = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fasth3-accel-nvenc")

    def __getitem__(self, key):
        if len(key) == 3:
            key = key + (slice(None), slice(None))
        if (len(key) != 5 or key[0] not in (slice(None), slice(0, 1))
                or any(key[i] != slice(None) for i in (1, 3, 4))):
            raise ValueError("ComfyUI's native H3 output-buffer contract changed.")
        if key[2].step is not None or key[2].start != self.written:
            raise ValueError("H3 chunks must be written sequentially.")
        self.pending = key[2].stop - key[2].start
        return self

    def copy_(self, part):
        mm.throw_exception_if_processing_interrupted()
        if part.shape[0] != 1 or part.shape[1] != 3 or part.shape[2] != self.pending:
            raise ValueError("Unexpected H3 finalized chunk shape.")
        frames = part[0].movedim(1, 0)  # [count, 3, h, w] in [0, 1]
        dtype = next(self.model.parameters()).dtype
        pieces = []
        with torch.inference_mode():
            for i in range(0, frames.shape[0], self.batch_frames):
                y = self.model(frames[i:i + self.batch_frames].to(dtype).contiguous(memory_format=torch.channels_last))
                if self.scale_by != 1.0:
                    y = F.interpolate(y, size=self.out_hw, mode=self.method, align_corners=False)
                top, left, crop_h, crop_w = self.crop
                y = y[..., top:top + crop_h, left:left + crop_w]
                pieces.append(y.clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(0, 2, 3, 1))
        pixels = torch.cat(pieces).cpu().numpy()
        if self.future is not None:
            self.future.result()
        self.future = self.worker.submit(self.encode, pixels)
        self.written += self.pending
        self.pending = None
        return self

    def encode(self, pixels):
        for pixels_frame in pixels:
            frame = av.VideoFrame.from_ndarray(pixels_frame, format="rgb24")
            frame = frame.reformat(format="yuv420p", dst_colorspace=1)
            color_srgb(frame)
            self.output.mux(self.stream.encode(frame))

    def finish(self):
        if self.future is not None:
            self.future.result()
        if self.written != self.frames:
            raise RuntimeError(f"Incomplete decode: {self.written}/{self.frames} frames.")
        self.output.mux(self.stream.encode(None))

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)


class FastH3AccelStreamSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",), "vae": ("VAE",), "upscale_model": ("UPSCALE_MODEL",),
            "scale_by": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.0001,
                                   "tooltip": "Resize after the x4 model: 0.5 = x2 (512x896 -> 1024x1792)."}),
            "method": (["bicubic", "bilinear"], {"default": "bicubic"}),
            "batch_frames": ("INT", {"default": 4, "min": 1, "max": 64}),
            "filename_prefix": ("STRING", {"default": "video/FastH3_OmniRef_Accel"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            "nvenc_gpu": ("INT", {"default": 0, "min": 0, "max": 31}),
            "cq": ("INT", {"default": 18, "min": 0, "max": 51}),
            "tile_overlap": ("INT", {"default": 64, "min": 16, "max": 128, "step": 16,
                                     "tooltip": "H3 VAE spatial tile overlap (native 64). 32 decodes ~20% faster at 512x896."}),
            "crop_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                                   "tooltip": "Center-crop the resized frames to this width (0 = no crop)."}),
            "crop_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                                    "tooltip": "Center-crop the resized frames to this height (0 = no crop)."}),
        }, "optional": {"audio": ("AUDIO",)}, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/FastH3 Accel"
    DESCRIPTION = "MiniMax H3 VAE decode, frame-batched model upscale + resize and async H.264 NVENC, overlapped per VAE chunk."

    def save(self, samples, vae, upscale_model, scale_by, method, batch_frames, filename_prefix, fps, nvenc_gpu, cq,
             tile_overlap=64, crop_width=0, crop_height=0, audio=None, prompt=None, extra_pnginfo=None):
        from comfy.cli_args import args
        if "h264_nvenc" not in av.codecs_available:
            raise RuntimeError("This PyAV/FFmpeg build lacks h264_nvenc.")
        video = samples["samples"]
        if getattr(video, "is_nested", False):
            video = video.unbind()[0]
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
            raise ValueError("Expected one H3 video latent [1, 24, T, H, W].")
        inner = vae.first_stage_model
        if not hasattr(inner, "decode_output_shape"):
            raise ValueError("Connect the MiniMax H3 video VAE.")
        rate = Fraction(str(fps)).limit_denominator(100000)

        mm.load_models_gpu([vae.patcher, upscale_model.patcher],
                           memory_required=vae.memory_used_decode(video.shape, vae.vae_dtype), force_full_load=True)
        shape = inner.decode_output_shape(video.shape)
        frames, height, width = shape[2], shape[3], shape[4]
        out_hw = (round(height * upscale_model.scale * scale_by), round(width * upscale_model.scale * scale_by))
        crop_h = min(crop_height or out_hw[0], out_hw[0])
        crop_w = min(crop_width or out_hw[1], out_hw[1])
        crop = ((out_hw[0] - crop_h) // 2, (out_hw[1] - crop_w) // 2, crop_h, crop_w)

        waveform, layout, sample_rate = None, None, None
        if audio is not None:
            sample_rate = int(audio["sample_rate"])
            data = audio["waveform"]
            waveform = data[0, :, :math.ceil(sample_rate / rate * frames)].float().cpu().contiguous().numpy()
            layout = "mono" if waveform.shape[0] == 1 else "stereo"

        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), crop_w, crop_h)
        filename = f"{name}_{counter:05}_.mp4"
        handle, temporary_name = tempfile.mkstemp(prefix=".fasth3-accel-", suffix=".part.mp4", dir=folder)
        os.close(handle)
        temporary = Path(temporary_name)
        model = upscale_model.model
        model.half().to(memory_format=torch.channels_last)  # ~20% faster convs, output within 1/255
        try:
            with av.open(str(temporary), "w", format="mp4") as output:
                if not args.disable_metadata and prompt is not None:
                    output.metadata["prompt"] = json.dumps(prompt)
                    for key, value in (extra_pnginfo or {}).items():
                        output.metadata[key] = json.dumps(value)
                stream = output.add_stream("h264_nvenc", rate=rate)
                stream.width, stream.height, stream.pix_fmt, stream.bit_rate = crop_w, crop_h, "yuv420p", 0
                stream.options = {"preset": "p4", "tune": "hq", "rc": "vbr", "cq": str(cq), "gpu": str(nvenc_gpu)}
                color_srgb(stream.codec_context)
                audio_stream = output.add_stream("aac", rate=sample_rate, layout=layout) if waveform is not None else None
                output.start_encoding()
                sink = UpscaleSink(output, stream, shape, model, scale_by, method, batch_frames, out_hw, crop)
                saved_overlap = inner.tile_overlap_min
                inner.tile_overlap_min = tile_overlap
                try:
                    with torch.inference_mode():
                        inner.decode(video.to(device=vae.device, dtype=vae.vae_dtype), output_buffer=sink)
                    sink.finish()
                finally:
                    inner.tile_overlap_min = saved_overlap
                    sink.close()
                if audio_stream is not None:
                    frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout=layout)
                    frame.sample_rate, frame.pts = sample_rate, 0
                    output.mux(audio_stream.encode(frame))
                    output.mux(audio_stream.encode(None))
            mm.throw_exception_if_processing_interrupted()
            os.replace(temporary, Path(folder) / filename)
        finally:
            model.float().to(memory_format=torch.contiguous_format)
            temporary.unlink(missing_ok=True)
        return {"ui": {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}], "animated": [True]}}


NODE_CLASS_MAPPINGS = {"FastH3AccelStreamSave": FastH3AccelStreamSave}
NODE_DISPLAY_NAME_MAPPINGS = {"FastH3AccelStreamSave": "FastH3 Accel Decode + Upscale + NVENC Save"}
