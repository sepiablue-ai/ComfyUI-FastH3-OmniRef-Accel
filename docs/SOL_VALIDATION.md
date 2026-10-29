# Sol VAE INT8 validation — 2026-10-29

## Result

| RTX 3060 12 GB | Original H3 INT8 VAE | Sol H3 INT8 VAE |
|---|---:|---:|
| Complete workflow | 126.444 s | 113.441 s |
| Decode + upscale + NVENC node | 22.001 s | 10.946 s |

Complete-workflow reduction: 13.003 s, 10.28%. One run per case, not an average. The author reviewed and accepted the Sol image quality for this input. No general perceptual-equivalence claim is made. Both videos decoded completely: 124 frames, 1024x1792, 24 fps (about 5.17 s), with identical decoded audio PCM.

## Conditions and measurement boundaries

- Windows, RTX 3060 12 GB (physical GPU 1, masked as CUDA device 0). Another workload on GPU 0 continued; CPU/storage contention may affect results. OS cache was not cleared.
- ComfyUI 0.43.0, official commit `0df1646a9cd71d4c990a87014fb37797a2956ead`. A fresh server process for each case; no node output cache hits, saved video input, or saved latent reuse.
- Time spans ComfyUI `execution_start` to `execution_success` after MP4 save. It includes model loading, conditioning, sampling, audio/video decode, upscale and saving. Server startup, input preparation and subsequent validation are excluded.
- The output-node call was timed with CUDA synchronization within each complete run. This is a component of the total, not an additional duration to add to it.
- Actual FastH3 model: `fasth3_omniref_8step_pruned_int8_convrot.safetensors`. The package's BF16 text encoder, prefetch, chunked FFN, sparse producer reuse and LBH continuation are active.
- First 5 steps at 256x448, LBH doubles the video latent, final 3 steps at 512x896. Same seed, prompt, reference, VSA, audio VAE, upscaler and encoder in both cases.
- Reference: the author's anime image `00033-604476799.webp`, prepared with the previously requested centered left/right crop and aspect-preserving resize to 544x960; no added padding. The image is not bundled. Published workflows use the portable placeholder `reference.png` and a different output prefix; generation settings are unchanged.
- Original repo example prompt with speech and background music, unchanged between cases.
- `realesr-animevideov3.pth`, FP16 channels-last, batch 4, x4 followed by bicubic 0.5; H.264 NVENC p4/hq CQ18, AAC 32 kHz. No AnimeSR is used in this variant.
- The normal H3 INT8 VAE still encodes the reference. A separate Sol VAELoader supplies only the final video decode node. Audio decode is unchanged.
- No repairs, text overlays, extra sharpening, audio replacement or other output edits were applied to the measured videos.
- Sol timings on RTX 4070 are unmeasured. The original 4070 results in VALIDATION.md are a separate benchmark.

## Stream compatibility

Sol writes finalized chunks using `output_buffer[0:1, :, start:end]`; the original H3 decoder uses five explicit axes. `UpscaleSink` now accepts both forms for batch 1 with full channel/spatial slices. Sequential time and chunk-shape checks remain active. VAE calculations and weights are unchanged. The same sink was used in both measured cases.

## Reproduction and checks

Use the Sol installation steps in the README and the bundled GUI/API workflow. Select your own reference image. Model licenses apply independently; no models are bundled or downloaded by setup.

The standard-library test suite checks both workflow graphs, preserves the original conditioning VAE and generation settings, and exercises both decoders' sink indexing contracts including tail chunks and rejected invalid slices. `setup.py --check --sol` additionally checks native nodes, NVENC availability, Sol decoder support and all seven model files without generation or installation.

Release checks passed: all 12 unittest cases, `git diff --check`, and `setup.py --check --sol` in the isolated ComfyUI environment (comfy-kitchen 0.2.40, PyAV 18.1.0). The setup-only `nodes_replacements.py` warning about a missing `PromptServer.instance` was present as documented in the README; required native nodes loaded and the check returned `check_passed`. The released `stream.py` is byte-identical to the file used for both complete runs. The original API workflow is unchanged.

Exact timings, audio PCM hash and the Sol weight SHA-256 are recorded in [sol_benchmark.json](sol_benchmark.json). Original model hashes and earlier RTX 4070 measurements remain in [VALIDATION.md](VALIDATION.md).
