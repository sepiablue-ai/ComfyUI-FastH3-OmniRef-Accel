# Validation (2026-10-23)

## Environment

Windows 11, RTX 4070 12 GB (driver WDDM, desktop on the same GPU), 48 GB RAM.
ComfyUI `1a82e6d0` (2026-10-21), Python 3.13.13, PyTorch 2.14.0+cu130, comfy-kitchen 0.2.39, PyAV 19.0.2.
Launch flags: `--disable-pinned-memory`, Dynamic VRAM (aimdo) on, `CUDA_VISIBLE_DEVICES=0`.

## Fresh install

A new ComfyUI checkout at `1a82e6d0` with its own uv venv (no pip) and the models shared through `extra_model_paths.yaml`:

1. `setup.bat COMFY PYTHON --check` → `check_passed` (7 nodes, six models found, h264_nvenc available).
2. `setup.bat COMFY PYTHON --install-dependencies` → `installed` (nothing missing, package copied to `custom_nodes`).
3. `workflows/fasth3_omniref_accel.api.json` queued through the API with the development prompt and reference image, seed 771204598.
4. The GUI workflow was produced by loading the API workflow in this ComfyUI's frontend; converting it back to the API format gave no input or link differences.
5. `python -m unittest discover -s tests` → 6 tests OK.

| Output size | Exec time (s) | Peak VRAM (MiB) | Result |
|---|---|---|---|
| 1024x1792 (first run after server start) | 66.6 | 11631 | bit-identical to the development pipeline |
| 1792x1024 | 61.7 | 11566 | bit-identical; 1792x1024x124 |
| 720x1280 | 61.8 | 11480 | bit-identical; generated at 512x896 (minimum short side), cropped to 720x1280x124 |

Exec time is ComfyUI `execution_start` → `execution_success`, server startup excluded. The 1792x1024 and 720x1280 runs followed on the same server
(models loaded, text encoder re-run). Peak RAM reads about 28 GB because the prefetched DiT file pages count toward the process working set.

## Speed history (1024x1792 output, 5 s, first run after server start)

| Configuration | Time (s) |
|---|---|
| Official template, VSA10, 1024x1792 direct | 459.6 |
| 512x896 generation + X2 VAE | 108.0 |
| + FFN chunks, low-res first 5 steps (LBH), streamed ESRGAN | 81.3 |
| + nvfp4 text encoder with BF16 compute, faster LBH load | 73.9–76.8 |
| + sparse-producer projection reuse, VAE tile overlap 32 | 70.2 |
| + DiT prefetch, ESRGAN channels_last (this pack) | 64.6–66.8 |

## Quality checks

- Lossless steps (prefetch, FFN chunks, producer reuse, LBH fast load, repackaging) were checked as bit-identical video and audio at the same seed.
- Lossy steps were checked against the previous configuration at the same seed: frames and face crops, plus the h3-audio-judge screen (SQUIM PESQ/STOI, L/R correlation) and listening:
  low-res first steps (k=2–5: OK on 2 seeds), BF16 text encoder (OK), nvfp4 text encoder (OK), VAE overlap 32 (Y-PSNR 38–41 dB, no visible seams, audio identical).
- Rejected: step reduction, FC1 W4A4 (visible detail loss), EasyCache (face collapse), X2 VAE (ESRGAN looked cleaner), 1344x768 / 1120x640 landscape modes (better small faces but judged not worth the time).
- Known limitation: small faces in wide shots are crushed by the 1/4-size first pass; the official direct generation keeps them.

Not validated: other GPUs, Linux, ComfyUI versions other than `1a82e6d0`, clips longer than 5 s with this exact configuration.

## Model files used

| File | SHA-256 |
|---|---|
| fasth3_omniref_8step_pruned_int8_convrot.safetensors | 851b2dc9670ed864b7f1e951ac9da047d83b1c04ded9bfcc1a9e5bba0ee33f80 |
| qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | 35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6 |
| minimax_h3_video_vae_int8_convrot.safetensors | 52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e |
| minimax_h3_audio_vae_fp32.safetensors | 8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48 |
| minimax_h3_latent_upscaler_3d_fp16.safetensors | 043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2 |
| realesr-animevideov3.pth | b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d |
