from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
VENDOR_DIR = RUNTIME_DIR / "InspireMusic"
MODELS_DIR = PROJECT_ROOT / ".models"
DEFAULT_CODE_REPO = os.getenv("INSPIREMUSIC_CODE_REPO", "https://github.com/FunAudioLLM/InspireMusic.git")
DEFAULT_MODEL_REPO = os.getenv("INSPIREMUSIC_MODEL_REPO", "FunAudioLLM/InspireMusic-Base-24kHz")
SKIP_VENDOR_PACKAGES = {
    "deepspeed",
    "diffusers",
    "flash-attn",
    "onnxruntime",
    "onnxruntime-gpu",
    "peft",
}


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def install_system_packages() -> None:
    if platform.system() != "Linux":
        return
    if "COLAB_GPU" not in os.environ and "google.colab" not in sys.modules:
        return

    run(["apt-get", "update"])
    run(["apt-get", "install", "-y", "git-lfs", "ffmpeg"])
    run(["git", "lfs", "install"])


def clone_vendor_repo(code_repo: str) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if VENDOR_DIR.exists():
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=VENDOR_DIR)
        patch_vendor_source()
        return
    run(["git", "clone", "--recursive", code_repo, str(VENDOR_DIR)])
    patch_vendor_source()


def patch_vendor_source() -> None:
    qwen_encoder = VENDOR_DIR / "inspiremusic" / "transformer" / "qwen_encoder.py"
    if qwen_encoder.exists():
        content = qwen_encoder.read_text(encoding="utf-8")
        patched = content.replace('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"')
        if patched != content:
            qwen_encoder.write_text(patched, encoding="utf-8")

    vqvae_file = VENDOR_DIR / "inspiremusic" / "music_tokenizer" / "vqvae.py"
    if vqvae_file.exists():
        content = vqvae_file.read_text(encoding="utf-8")
        patched = content.replace("ckpt = torch.load(ckpt_path)", "ckpt = torch.load(ckpt_path, map_location='cpu')")
        if patched != content:
            vqvae_file.write_text(patched, encoding="utf-8")

    processor_file = VENDOR_DIR / "inspiremusic" / "dataset" / "processor.py"
    if processor_file.exists():
        content = processor_file.read_text(encoding="utf-8")
        patched = content.replace(
            "if hasattr(torchaudio, 'set_audio_backend'):\n    torchaudio.set_audio_backend('soundfile')",
            "# torchaudio.set_audio_backend('soundfile')",
        )
        patched = patched.replace(
            "torchaudio.set_audio_backend('soundfile')",
            "# torchaudio.set_audio_backend('soundfile')",
        )
        if patched != content:
            processor_file.write_text(patched, encoding="utf-8")



def patch_model_yaml(model_dir: Path) -> None:
    yaml_path = model_dir / "inspiremusic.yaml"
    if not yaml_path.exists():
        return

    content = yaml_path.read_text(encoding="utf-8")
    replacements = {
        "../../": "",
        "pretrained_models/InspireMusic-Base-24kHz/": f"{model_dir.as_posix()}/",
        "pretrained_models/InspireMusic-Base-24kHz/music_tokenizer": f"{(model_dir / 'music_tokenizer').as_posix()}",
    }
    patched = content
    for source, target in replacements.items():
        patched = patched.replace(source, target)
    if patched != content:
        yaml_path.write_text(patched, encoding="utf-8")


def _build_filtered_vendor_requirements() -> Path:
    source = VENDOR_DIR / "requirements.txt"
    if not source.exists():
        raise SystemExit("Upstream InspireMusic requirements.txt is missing.")

    filtered_lines: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        package_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
        if package_name in SKIP_VENDOR_PACKAGES:
            continue
        filtered_lines.append(line)

    filtered_lines.extend(
        [
            "diffusers==0.27.2",
            "huggingface-hub>=0.30.0,<1.0.0",
            "peft>=0.17.0,<1.0.0",
            "protobuf>=5.26.1,<6.0.0",
            "transformers==4.46.3",
        ]
    )

    handle = tempfile.NamedTemporaryFile("w", suffix="-opt-music-vendor.txt", delete=False, encoding="utf-8")
    with handle:
        handle.write("\n".join(filtered_lines))
        handle.write("\n")
    return Path(handle.name)


def install_python_packages() -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")])

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torchao",
            "transformers",
            "diffusers",
            "peft",
        ]
    )
    filtered_requirements = _build_filtered_vendor_requirements()
    run([sys.executable, "-m", "pip", "install", "-r", str(filtered_requirements)])


def download_model(model_repo: str) -> None:
    from huggingface_hub import snapshot_download

    model_name = model_repo.rsplit("/", 1)[-1]
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if not any(model_dir.iterdir()):
        snapshot_download(repo_id=model_repo, local_dir=str(model_dir))

    patch_model_yaml(model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Opt Music for Colab or local use.")
    parser.add_argument("--code-repo", default=DEFAULT_CODE_REPO)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--skip-system", action="store_true")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if shutil.which("git") is None:
        raise SystemExit("git is required before running bootstrap.py")

    if not args.skip_system:
        install_system_packages()

    clone_vendor_repo(args.code_repo)

    if not args.skip_python:
        install_python_packages()

    if not args.skip_model:
        download_model(args.model_repo)

    print("Bootstrap complete.")
    print("Next: python app.py --share")


if __name__ == "__main__":
    main()
