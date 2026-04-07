from __future__ import annotations

import re
from pathlib import Path

import lameenc
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


TARGET_LUFS = -18.0


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "track"


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array.mean(axis=1 if array.shape[1] <= 2 else 0).astype(np.float32)
    raise ValueError("Audio array must be 1D or 2D.")


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(Path(path), always_2d=False)
    return ensure_mono(audio), int(sample_rate)


def apply_edge_fades(audio: np.ndarray, sample_rate: int, fade_ms: int = 120) -> np.ndarray:
    waveform = ensure_mono(audio).copy()
    fade_samples = max(1, int(sample_rate * fade_ms / 1000))
    fade_samples = min(fade_samples, len(waveform) // 2)
    if fade_samples == 0:
        return waveform

    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    waveform[:fade_samples] *= fade_in
    waveform[-fade_samples:] *= fade_out
    return waveform


def normalize_background_loudness(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = TARGET_LUFS,
) -> np.ndarray:
    waveform = ensure_mono(audio)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if not np.any(waveform):
        return waveform

    meter = pyln.Meter(sample_rate)
    try:
        measured_loudness = meter.integrated_loudness(waveform)
        normalized = pyln.normalize.loudness(waveform, measured_loudness, target_lufs)
    except Exception:
        rms = float(np.sqrt(np.mean(np.square(waveform))) + 1e-8)
        target_rms = 10 ** (target_lufs / 20.0)
        normalized = waveform * (target_rms / rms)

    peak = float(np.max(np.abs(normalized)) + 1e-8)
    if peak > 0.98:
        normalized = normalized * (0.98 / peak)

    return normalized.astype(np.float32)


def export_wav(audio: np.ndarray, sample_rate: int, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, ensure_mono(audio), sample_rate, subtype="PCM_16")
    return str(path)


def export_mp3(audio: np.ndarray, sample_rate: int, output_path: str | Path, bitrate_kbps: int = 192) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pcm = np.clip(ensure_mono(audio), -1.0, 1.0)
    pcm_int16 = (pcm * 32767.0).astype(np.int16)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)

    mp3_bytes = encoder.encode(pcm_int16.tobytes())
    mp3_bytes += encoder.flush()

    with path.open("wb") as handle:
        handle.write(mp3_bytes)

    return str(path)
