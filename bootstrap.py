from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
VENDOR_DIR = RUNTIME_DIR / "InspireMusic"
MODELS_DIR = PROJECT_ROOT / ".models"
DEFAULT_CODE_REPO = os.getenv("INSPIREMUSIC_CODE_REPO", "https://github.com/FunAudioLLM/InspireMusic.git")
DEFAULT_MODEL_REPO = os.getenv("INSPIREMUSIC_MODEL_REPO", "FunAudioLLM/InspireMusic-Base-24kHz")


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
        return
    run(["git", "clone", "--recursive", code_repo, str(VENDOR_DIR)])


def install_python_packages() -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")])

    vendor_requirements = VENDOR_DIR / "requirements.txt"
    if vendor_requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(vendor_requirements)])

    run([sys.executable, "-m", "pip", "install", "-e", str(VENDOR_DIR)])


def download_model(model_repo: str) -> None:
    from huggingface_hub import snapshot_download

    model_name = model_repo.rsplit("/", 1)[-1]
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if not any(model_dir.iterdir()):
        snapshot_download(repo_id=model_repo, local_dir=str(model_dir))

    yaml_path = model_dir / "inspiremusic.yaml"
    if yaml_path.exists():
        content = yaml_path.read_text(encoding="utf-8")
        patched = content.replace("../../", "")
        if patched != content:
            yaml_path.write_text(patched, encoding="utf-8")


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
