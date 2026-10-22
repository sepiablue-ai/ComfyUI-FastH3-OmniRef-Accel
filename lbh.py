"""LBH upscale of the clean low-resolution video, CONST re-noise of the video, audio carried over.

Adapted from sepiablue-ai's ComfyUI-H3-LBH-X2-Stream handoff.py (GPL-3.0). The upscaler is the
unchanged LBH-123-AI implementation in vendor/lbh_3d.py (MIT, see NOTICE.md). Its own loader builds
the ~345M-parameter network on the CPU in FP32 with random initialization before loading the
checkpoint (about 1.3 s); _preload builds it on the meta device, assigns the FP16 checkpoint on the
GPU and stores it in the loader's own cache, so the vendored upscale call is otherwise unchanged.
"""
import gc

import torch

import comfy.model_management as mm
from comfy.nested_tensor import NestedTensor
import folder_paths

from .vendor import lbh_3d as lbh


def _preload(path, device):
    key = f"{path}::{lbh._backend_label(device)}::fp16"
    if key in lbh.MODEL_CACHE:
        return
    sd = lbh._extract_upscaler_sd(lbh._load_raw_sd(path))
    cfg = lbh._detect_arch(sd)
    with torch.device("meta"):
        model = lbh.LatentResizer3D(
            in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
            channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
            temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"])
    model.load_state_dict({k: v.to(device=device, dtype=torch.float16) for k, v in sd.items()}, strict=True, assign=True)
    if any(t.is_meta for t in [*model.parameters(), *model.buffers()]):
        raise RuntimeError("LBH module has tensors outside the checkpoint; cannot build on meta.")
    lbh.MODEL_CACHE[key] = model.eval().requires_grad_(False)


def renoise_video(model, noise, sigmas, latent):
    video, audio = latent["samples"].unbind()
    mask = NestedTensor([
        torch.ones((video.shape[0], 1, *video.shape[2:]), device=video.device, dtype=torch.float32),
        torch.zeros((audio.shape[0], 1, *audio.shape[2:]), device=audio.device, dtype=torch.float32),
    ])
    masked = {**latent, "noise_mask": mask}
    generated = list(noise.generate_noise(masked).unbind())
    generated[1] = torch.zeros_like(generated[1])
    generated = [member * member_mask.to(device=member.device, dtype=member.dtype).expand_as(member)
                 for member, member_mask in zip(generated, mask.unbind())]
    samples = latent["samples"]
    if any(torch.count_nonzero(member) > 0 for member in samples.unbind()):
        samples = model.get_model_object("process_latent_in")(samples)
    sampling = model.get_model_object("model_sampling")
    mixed = []
    for member, member_noise in zip(samples.unbind(), generated):
        current = sampling.noise_scaling(sigmas[0], member_noise, member)
        current = sampling.inverse_noise_scaling(sigmas[0], current)
        mixed.append(current)
    samples = model.get_model_object("process_latent_out")(NestedTensor(mixed))
    samples = NestedTensor([torch.nan_to_num(member, nan=0.0, posinf=0.0, neginf=0.0)
                            for member in samples.unbind()])
    result = {**masked, "samples": samples.cpu(), "noise_mask": mask.cpu()}
    gc.collect()
    mm.soft_empty_cache()
    return result


class FastH3AccelLBHHandoff:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",), "model": ("MODEL",), "noise": ("NOISE",), "sigmas": ("SIGMAS",),
            "upscaler_name": (folder_paths.get_filename_list("latent_upscale_models"),),
            "width": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 896, "min": 32, "max": 8192, "step": 32}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "MiniMax H3/FastH3 Accel"
    DESCRIPTION = ("Upscale the low sampler's clean video (denoised_output) with LBH to width x height and re-noise it "
                   "for the remaining sigmas. Concatenate the low sampler's intermediate audio with the result.")

    def upscale(self, samples, model, noise, sigmas, upscaler_name, width, height):
        if len(sigmas) < 2 or sigmas.device.type != "cpu":
            raise ValueError("Connect the remaining original CPU sigma schedule from SplitSigmas")
        video, audio = samples["samples"].unbind()
        path = folder_paths.get_full_path_or_raise("latent_upscale_models", upscaler_name)
        device = lbh._resolve_device("cuda")
        _preload(path, device)
        mm.free_memory(4 * 1024**3, device)
        resized = lbh.MinimaxH3LatentUpscaler3D.execute(
            latent={"samples": video}, model_name=path,
            mode={"mode": lbh.UpscaleMode.TARGET_DIMENSIONS, "width": width, "height": height},
            align=32, enable_chunking=False, device="cuda", precision="fp16",
        )[0]["samples"]
        if tuple(resized.shape[-2:]) != (height // 16, width // 16):
            raise ValueError("LBH target dimensions must align to the H3 32px grid")
        upscaled = samples.copy()
        upscaled.pop("noise_mask", None)
        upscaled["samples"] = NestedTensor([resized, audio])
        return (renoise_video(model, noise, sigmas, upscaled),)


NODE_CLASS_MAPPINGS = {"FastH3AccelLBHHandoff": FastH3AccelLBHHandoff}
NODE_DISPLAY_NAME_MAPPINGS = {"FastH3AccelLBHHandoff": "FastH3 Accel LBH Upscale + Continue"}
