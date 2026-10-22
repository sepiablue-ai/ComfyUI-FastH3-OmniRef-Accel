"""Use an existing ComfyUI Python; no ComfyUI launch, model download or core edits."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODELS = {
    "diffusion_models": ["fasth3_omniref_8step_pruned_int8_convrot.safetensors"],
    "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
    "video_vae": ["minimax_h3_video_vae_int8_convrot.safetensors"],
    "audio_vae": ["minimax_h3_audio_vae_fp32.safetensors"],
    "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors",
                              "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"],
    "upscale_models": ["realesr-animevideov3.pth"],
}
NATIVE_NODES = ["MiniMaxH3ReferenceToVideo", "MiniMaxH3SigmaShift", "EmptyMiniMaxH3LatentAV", "BlockSparseAttention",
                "ModelAttentionBackend", "LTXVSeparateAVLatent", "LTXVConcatAVLatent", "UpscaleModelLoader"]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def package_files():
    excluded = {".git", "__pycache__", ".pytest_cache", "tmp"}
    return [path for path in ROOT.rglob("*") if path.is_file()
            and not excluded.intersection(path.relative_to(ROOT).parts)
            and path.suffix not in {".pyc", ".mp4", ".safetensors", ".pth"}]


def install_package(comfy_root):
    destination = comfy_root / "custom_nodes" / ROOT.name
    if destination.resolve() == ROOT.resolve():
        return destination
    files = package_files()
    if destination.exists():
        for source in files:
            existing = destination / source.relative_to(ROOT)
            if not existing.is_file() or sha(existing) != sha(source):
                raise RuntimeError("Installed package differs; preserve it and install this checkout manually: " + str(destination))
        return destination
    destination.mkdir(parents=True, exist_ok=False)
    for source in files:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def install_requirements():
    """Install requirements.txt into this Python only when something is missing (uv venvs have no pip)."""
    from importlib.metadata import PackageNotFoundError, version
    missing = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        requirement = line.split("#")[0].strip()
        if not requirement:
            continue
        name, _, minimum = requirement.partition(">=")
        try:
            installed = tuple(int(part) for part in version(name).split(".")[:2])
            wanted = tuple(int(part) for part in minimum.split(".")[:2]) if minimum else ()
            if installed < wanted:
                missing.append(requirement)
        except PackageNotFoundError:
            missing.append(requirement)
    if not missing:
        return
    if importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", "install", "--no-deps", *missing]
    elif shutil.which("uv"):
        command = ["uv", "pip", "install", "--python", sys.executable, "--no-deps", *missing]
    else:
        raise RuntimeError("Neither pip nor uv is available for this Python; install manually: " + " ".join(missing))
    subprocess.run(command, check=True)


def import_package(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy_argument", nargs="?", type=Path, help="ComfyUI folder passed by setup.bat")
    parser.add_argument("python_argument", nargs="?", help="Python selected by setup.bat")
    parser.add_argument("--comfy-root", type=Path)
    parser.add_argument("--check", action="store_true", help="Read-only environment and model check")
    parser.add_argument("--install-dependencies", action="store_true", help="Install requirements.txt into this ComfyUI's Python")
    parser.add_argument("--extra-model-paths", type=Path, help="Additional native ComfyUI model-path YAML")
    args = parser.parse_args()
    if args.check and args.install_dependencies:
        parser.error("--check cannot be combined with installation")
    selected_root = args.comfy_root or args.comfy_argument or os.environ.get("FASTH3_ACCEL_SETUP_ROOT")
    if not selected_root:
        parser.error("Specify the ComfyUI folder")
    comfy_root = Path(selected_root).resolve()
    if not (comfy_root / "main.py").is_file() or not (comfy_root / "comfy/sd.py").is_file():
        parser.error("--comfy-root must contain ComfyUI main.py and comfy/sd.py")
    portable = Path(sys.executable).resolve().parent.name.lower() == "python_embeded"
    if sys.prefix == sys.base_prefix and not portable:
        parser.error("Use the existing ComfyUI venv or portable Python; global/system Python is refused")
    if args.install_dependencies:
        install_requirements()
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(comfy_root))
    sys.argv = [sys.argv[0], "--disable-pinned-memory"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths
    from utils.extra_config import load_extra_path_config
    config = comfy_root / "extra_model_paths.yaml"
    if config.is_file():
        load_extra_path_config(str(config))
    if args.extra_model_paths:
        load_extra_path_config(str(args.extra_model_paths.resolve()))
    import av
    if "h264_nvenc" not in av.codecs_available:
        raise RuntimeError("This PyAV/FFmpeg build lacks h264_nvenc; the save node needs NVIDIA NVENC")
    import nodes
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    missing = [name for name in NATIVE_NODES if name not in nodes.NODE_CLASS_MAPPINGS]
    if missing:
        raise RuntimeError("This ComfyUI lacks native nodes " + ", ".join(missing) + "; update ComfyUI")
    package = import_package("fasth3_accel_setup_package")
    found = {}
    for category, filenames in MODELS.items():
        folder = "vae" if category.endswith("vae") else category
        candidates = folder_paths.get_filename_list(folder)
        matches = [name for name in candidates if Path(name).name in filenames]
        if not matches:
            raise FileNotFoundError(f"Model missing from {folder}: {filenames[0]}; see README.md")
        found[category] = folder_paths.get_full_path_or_raise(folder, matches[0])
    installed = str(comfy_root / "custom_nodes" / ROOT.name) if args.check else str(install_package(comfy_root))
    import importlib.metadata
    print(json.dumps({"status": "check_passed" if args.check else "installed", "python": sys.executable,
        "comfy_root": str(comfy_root), "package": installed, "nodes": sorted(package.NODE_CLASS_MAPPINGS),
        "comfy-kitchen": importlib.metadata.version("comfy-kitchen"), "pyav": av.__version__, "models": found,
        "video_generations": 0, "model_downloads": 0, "core_files_changed": False}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
