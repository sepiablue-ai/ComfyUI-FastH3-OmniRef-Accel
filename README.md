# ComfyUI FastH3 OmniRef Accel

English | [日本語](README_ja.md)

> **Unofficial** nodes. Not affiliated with FastVideo, Hao AI Lab or Nuva Lab. FastH3 OmniRef weights are not included.

ComfyUI nodes and a workflow that run FastH3 OmniRef (the 8-step distilled MiniMax H3 Ref2VA model) much faster on an RTX 4070 12 GB.
The 8 sampling steps are kept; the time goes away in the generation size, the text encoder and the decode path.

| 5 s, 124 frames, with audio, first run after server start | Time |
|---|---|
| Official template (VSA10, 1024x1792 generated directly) | 459.6 s |
| **This pack (1024x1792 output)** | **64.6–66.8 s (about 7x)** |

Later runs on the same server with a new prompt take about 62 s. Measured on an RTX 4070 12 GB with 48 GB RAM ([docs/VALIDATION.md](docs/VALIDATION.md)).

## Known limitation (image quality)

- **Small faces lose quality** (wide landscape shots, full body, distant subjects). Faces that the official direct generation keeps can be crushed here,
  because the first 5 steps run at 1/4 of the output size, where such a face is only a few latent cells, and the last steps do not restore it.
- Intended for portrait character videos framed from bust-up to knee-up.
- The generation size differs from the official setup, so the same seed gives different motion and composition.

## How it works

1. **FastH3 Accel Sizes** takes the final output size and returns the generation size (1/2 of the output, short side at least `min_short_side`), the first-pass size (half of that), the post-ESRGAN resize and the crop size.
2. The first 5 steps run at the low size, LBH's 3D latent upscaler doubles the clean video latent, and the last 3 steps run at the generation size. Audio continues through all 8 steps.
3. The H3 VAE decodes, Real-ESRGAN animevideov3 (x4) upscales, then a resize and a center crop give exactly the requested size without stretching.

| Node | What it does | Effect on output |
|---|---|---|
| FastH3 Accel Sizes (from output size) | Sizes from the output size; small outputs raise the generation short side to `min_short_side` (default 512) | - |
| FastH3 Accel Text Encoder BF16 | ComfyUI runs the text encoder LLM in FP32; this runs it in BF16 | Conditioning changes slightly (looks the same at the same seed) |
| FastH3 Accel Prefetch DiT Weights | Faults in the memory-mapped DiT on a background thread while the text encoder runs (hides about 6 s on the first run) | None |
| FastH3 Accel Chunk FeedForward | Splits the FFN rows into 8 chunks (about 10 s faster over 8 steps at 512x896) | None |
| FastH3 Accel Sparse Producer Reuse | The native sparse attention projects QKV and the VSA gate twice on each sampler's first step; reuse the first pass | None |
| FastH3 Accel LBH Upscale + Continue | LBH latent upscale, CONST re-noise of the video, audio state carried over; faster LBH load | None |
| FastH3 Accel Decode + Upscale + NVENC Save | ESRGAN, resize, crop and NVENC overlap per VAE chunk; VAE tile overlap 32 px | Y-PSNR 38–41 dB vs overlap 64 (no visible difference) |

"None" means bit-identical output at the same seed was verified.

The text encoder is the nvfp4_awq file (15.7 GB). FP4 math gives nothing on an RTX 4070, but the int8 file (26 GB) plus the 21 GB DiT
do not fit together in a 48 GB RAM file cache, so the text encoder was re-read from disk on every run.

## Requirements

- Windows, NVIDIA GPU (validated on an RTX 4070 12 GB), 48 GB RAM validated
- ComfyUI with native MiniMax H3 (Reference to Video) and BlockSparseAttention. Validated: ComfyUI `1a82e6d0` (2026-10-21), Python 3.13.13, PyTorch 2.14.0+cu130, comfy-kitchen 0.2.39, PyAV 19.0.2
- PyAV with h264_nvenc (ComfyUI's PyAV works)
- ComfyUI Dynamic VRAM left enabled; validated with `--disable-pinned-memory`

## Models

No weights are bundled or downloaded. Check each source's terms.

| File | Folder | Source |
|---|---|---|
| fasth3_omniref_8step_pruned_int8_convrot.safetensors | models/diffusion_models | [FastVideo/FastH3-Omni-Comfy](https://huggingface.co/FastVideo/FastH3-Omni-Comfy) |
| qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | models/text_encoders | same |
| minimax_h3_video_vae_int8_convrot.safetensors | models/vae | same |
| minimax_h3_audio_vae_fp32.safetensors | models/vae | same |
| minimax_h3_latent_upscaler_3d_fp16.safetensors | models/latent_upscale_models | [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) |
| realesr-animevideov3.pth | models/upscale_models | [xinntao/Real-ESRGAN v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0) |

The current LBH file name `minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors` is also accepted by the setup check; whether it equals the validated weights is unverified.
SHA-256 of the validated files is in [docs/VALIDATION.md](docs/VALIDATION.md).

## Install

Stop ComfyUI and run `setup.bat` from this folder. Inside `ComfyUI/custom_nodes` it detects ComfyUI and its Python.

```bat
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --check
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe"
```

- The second argument is **ComfyUI's own Python**. Portable `python_embeded` and `.venv` / `venv` inside or next to ComfyUI are searched; more than one candidate needs it explicitly. Global Python is refused.
- `--check` is read-only: native nodes, h264_nvenc and the six models.
- Without `--check` the package is copied to `ComfyUI/custom_nodes/ComfyUI-FastH3-OmniRef-Accel`; an existing folder with different content is never overwritten.
- No extra Python packages are needed (the PyAV it uses is already a ComfyUI dependency).
- Shared model folders use ComfyUI's `extra_model_paths.yaml`; `--extra-model-paths PATH` adds one for setup only.
- An `IMPORT FAILED: nodes_replacements.py` warning during setup comes from loading nodes without a server and does not affect generation.

## Use

1. Drag `workflows/fasth3_omniref_accel.json` into ComfyUI (API format: `fasth3_omniref_accel.api.json`).
2. Set the final size in **Output size** (FastH3 Accel Sizes): `output_width` / `output_height`.
3. Set the image in **Reference image** and the prompt in **Reference to Video (prompt)**, then queue. The prompt refers to the image as `<Picture 1>` (see `workflows/example_prompt.txt`).

| Output | First → last pass | Note |
|---|---|---|
| 1024x1792 | 256x448 → 512x896 | validated reference |
| 1792x1024 | 448x256 → 896x512 | faces in wide shots degrade |
| 720x1280 | 256x448 → 512x896 | half would be 360x640, so the short side is raised to 512; 731x1280 is center-cropped |

Videos are saved as `output/video/FastH3_OmniRef_Accel_*.mp4` (H.264 NVENC CQ18, AAC).

## Notes

- Low-resolution steps are SplitSigmas `step` (default 5). 6 was fine for portrait close-ups (about 5 s faster), but gave an audio NG in a landscape test, so the default stays 5.
- The prefetch node makes the DiT file pages count toward the process, so Task Manager shows about 21 GB more RAM (the pages were already in the OS file cache).
- One video per execution (batch 1).

## License

Code: GPL-3.0 ([LICENSE](LICENSE)). Sources of bundled and adapted code are in [NOTICE.md](NOTICE.md).
Model weights keep their own terms (FastH3 OmniRef: MiniMax-H3 Community License; LBH and Real-ESRGAN: their distribution pages).
