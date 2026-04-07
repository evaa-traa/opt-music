from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import gettempdir
from typing import Dict
from uuid import uuid4

import numpy as np
import torch
import huggingface_hub
from huggingface_hub import snapshot_download

from utils import TARGET_LUFS, apply_edge_fades, export_mp3, export_wav, load_audio, normalize_background_loudness, slugify


DEFAULT_MODEL_REPO = os.getenv("INSPIREMUSIC_MODEL_REPO", "FunAudioLLM/InspireMusic-Base-24kHz")
DEFAULT_CODE_REPO = os.getenv("INSPIREMUSIC_CODE_REPO", "https://github.com/FunAudioLLM/InspireMusic.git")
SUPPORTED_DURATIONS = (15, 30)
SUPPORTED_CHORUS = ("intro", "verse", "chorus", "outro")

PRESET_PROMPTS: Dict[str, str] = {
    "Epic History": (
        "epic historical instrumental soundtrack, cinematic strings, taiko drums, brass swells, "
        "ancient empire mood, dramatic but controlled background energy"
    ),
    "Phonk": (
        "dark phonk instrumental, gritty bass, punchy cowbells, crisp drums, street racing energy, "
        "moody background groove"
    ),
    "Trailer": (
        "cinematic trailer instrumental, hybrid orchestra, deep percussion, tense risers, bold impacts, "
        "modern blockbuster background bed"
    ),
    "Documentary": (
        "documentary instrumental background music, warm piano, subtle strings, light percussion, "
        "inspiring and unobtrusive storytelling mood"
    ),
}

PROMPT_SUFFIX = (
    "instrumental only, no vocals, no singer, no speech, no narration, no rap, no lyrics, "
    "youtube shorts background music, clean intro, polished mix, stable groove, unobtrusive arrangement"
)


@dataclass
class GenerationResult:
    prompt: str
    wav_path: str
    mp3_path: str
    seed: int
    duration_seconds: int
    chorus: str
    device: str
    elapsed_seconds: float
    status_text: str


class RuntimeSetupError(RuntimeError):
    pass


def _patch_huggingface_hub_compat() -> None:
    if hasattr(huggingface_hub, "cached_download"):
        return

    def _cached_download(*args, **kwargs):
        return huggingface_hub.hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = _cached_download


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _runtime_root() -> Path:
    return _project_root() / ".runtime"


def _vendor_root() -> Path:
    return _runtime_root() / "InspireMusic"


def _models_root() -> Path:
    return _project_root() / ".models"


def _output_root() -> Path:
    return Path(gettempdir()) / "opt_music_outputs"


def _model_local_name(model_repo: str) -> str:
    return model_repo.rsplit("/", 1)[-1]


def _patch_model_yaml(model_dir: Path) -> None:
    yaml_path = model_dir / "inspiremusic.yaml"
    if not yaml_path.exists():
        return

    original = yaml_path.read_text(encoding="utf-8")
    model_name = model_dir.name
    patched = original.replace("../../", "")
    patched = patched.replace(
        f"pretrained_models/{model_name}/",
        f"{model_dir.as_posix()}/",
    )
    patched = patched.replace(
        f"pretrained_models/{model_name}/music_tokenizer",
        f"{(model_dir / 'music_tokenizer').as_posix()}",
    )
    if patched != original:
        yaml_path.write_text(patched, encoding="utf-8")


def _patch_vendor_source(vendor_dir: Path) -> None:
    qwen_encoder = vendor_dir / "inspiremusic" / "transformer" / "qwen_encoder.py"
    if not qwen_encoder.exists():
        qwen_encoder = None

    if qwen_encoder is not None:
        original = qwen_encoder.read_text(encoding="utf-8")
        patched = original.replace('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"')
        if patched != original:
            qwen_encoder.write_text(patched, encoding="utf-8")

    vqvae_file = vendor_dir / "inspiremusic" / "music_tokenizer" / "vqvae.py"
    if vqvae_file.exists():
        original = vqvae_file.read_text(encoding="utf-8")
        patched = original.replace("ckpt = torch.load(ckpt_path)", "ckpt = torch.load(ckpt_path, map_location='cpu')")
        if patched != original:
            vqvae_file.write_text(patched, encoding="utf-8")


def ensure_code_checkout(code_repo: str = DEFAULT_CODE_REPO) -> Path:
    vendor_dir = _vendor_root()
    vendor_dir.parent.mkdir(parents=True, exist_ok=True)

    if vendor_dir.exists():
        _patch_vendor_source(vendor_dir)
        return vendor_dir

    try:
        subprocess.run(
            ["git", "clone", "--recursive", code_repo, str(vendor_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeSetupError("git is required. Run bootstrap.py in Colab before launching the app.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeSetupError(exc.stderr.strip() or "Failed to clone InspireMusic repository.") from exc

    _patch_vendor_source(vendor_dir)
    return vendor_dir


def ensure_model_snapshot(model_repo: str = DEFAULT_MODEL_REPO) -> Path:
    model_dir = _models_root() / _model_local_name(model_repo)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not any(model_dir.iterdir()):
        snapshot_download(repo_id=model_repo, local_dir=str(model_dir))

    _patch_model_yaml(model_dir)
    return model_dir


@lru_cache(maxsize=1)
def _load_inspiremusic_classes():
    _patch_huggingface_hub_compat()
    vendor_dir = ensure_code_checkout()
    matcha_dir = vendor_dir / "third_party" / "Matcha-TTS"

    for path in (vendor_dir, matcha_dir):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        from inspiremusic.cli.inference import InspireMusicModel, env_variables
    except Exception as exc:  # pragma: no cover - dependency resolution failure path
        raise RuntimeSetupError(
            "InspireMusic dependencies are missing. Run `python bootstrap.py` before launching the app."
        ) from exc

    return InspireMusicModel, env_variables


class InspireMusicGenerator:
    def __init__(self, model_repo: str = DEFAULT_MODEL_REPO) -> None:
        self.model_repo = model_repo
        self.model_name = _model_local_name(model_repo)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = _output_root()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vendor_dir = ensure_code_checkout()
        self.model_dir = ensure_model_snapshot(model_repo)
        self.InspireMusicModel, self.env_variables = _load_inspiremusic_classes()
        self.env_variables()
        self.model = self.InspireMusicModel(
            model_name=self.model_name,
            model_dir=str(self.model_dir),
            min_generate_audio_seconds=10.0,
            max_generate_audio_seconds=float(max(SUPPORTED_DURATIONS)),
            sample_rate=24000,
            output_sample_rate=24000,
            load_jit=True,
            load_onnx=False,
            fast=True,
            result_dir=str(self.output_dir),
        )

    def _build_prompt(self, prompt: str) -> str:
        cleaned = " ".join(prompt.strip().split())
        if not cleaned:
            cleaned = PRESET_PROMPTS["Epic History"]
        return f"{cleaned}, {PROMPT_SUFFIX}"

    def generate(
        self,
        prompt: str,
        duration_seconds: int,
        seed: int,
        chorus: str,
        target_lufs: float = TARGET_LUFS,
    ) -> GenerationResult:
        if duration_seconds not in SUPPORTED_DURATIONS:
            raise ValueError(f"Duration must be one of {SUPPORTED_DURATIONS}.")
        if chorus not in SUPPORTED_CHORUS:
            raise ValueError(f"Chorus must be one of {SUPPORTED_CHORUS}.")

        started_at = time.perf_counter()
        full_prompt = self._build_prompt(prompt)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        stem = f"{slugify(prompt)[:48]}-{duration_seconds}s-{seed}-{uuid4().hex[:8]}"
        with torch.no_grad():
            raw_output_path = self.model.inference(
                task="text-to-music",
                text=full_prompt,
                audio_prompt=None,
                chorus=chorus,
                time_start=0.0,
                time_end=float(duration_seconds),
                output_fn=stem,
                max_audio_prompt_length=5.0,
                fade_out_duration=1.0,
                output_format="wav",
                fade_out_mode=True,
                trim=False,
            )

        raw_path = Path(raw_output_path)
        if not raw_path.is_absolute():
            raw_path = self.vendor_dir / raw_path
        if not raw_path.exists():
            raw_path = self.output_dir / f"{stem}.wav"
        if not raw_path.exists():
            raise RuntimeError("InspireMusic did not produce the expected WAV output.")

        waveform, sample_rate = load_audio(raw_path)
        waveform = apply_edge_fades(waveform, sample_rate)
        waveform = normalize_background_loudness(waveform, sample_rate, target_lufs=target_lufs)
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)

        wav_path = export_wav(waveform, sample_rate, self.output_dir / f"{stem}-normalized.wav")
        mp3_path = export_mp3(waveform, sample_rate, self.output_dir / f"{stem}.mp3")

        elapsed_seconds = time.perf_counter() - started_at
        status_text = (
            f"Prompt: {full_prompt}\n\n"
            f"Model: {self.model_repo} | Device: {self.device.upper()} | Duration: {duration_seconds}s | "
            f"Section: {chorus} | Seed: {seed} | Target loudness: {target_lufs:.0f} LUFS | "
            f"Elapsed: {elapsed_seconds:.1f}s"
        )

        return GenerationResult(
            prompt=full_prompt,
            wav_path=wav_path,
            mp3_path=mp3_path,
            seed=seed,
            duration_seconds=duration_seconds,
            chorus=chorus,
            device=self.device,
            elapsed_seconds=elapsed_seconds,
            status_text=status_text,
        )


@lru_cache(maxsize=1)
def get_generator(model_repo: str = DEFAULT_MODEL_REPO) -> InspireMusicGenerator:
    return InspireMusicGenerator(model_repo=model_repo)


def generate_music(
    prompt: str,
    duration_seconds: int,
    seed: int = 0,
    chorus: str = "intro",
) -> GenerationResult:
    generator = get_generator()
    return generator.generate(
        prompt=prompt,
        duration_seconds=int(duration_seconds),
        seed=int(seed),
        chorus=chorus,
    )
