from __future__ import annotations

import argparse

from model import PRESET_PROMPTS, SUPPORTED_CHORUS, SUPPORTED_DURATIONS, generate_music


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one instrumental track with InspireMusic.")
    parser.add_argument("--prompt", default=PRESET_PROMPTS["Epic History"])
    parser.add_argument("--duration", type=int, choices=SUPPORTED_DURATIONS, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chorus", choices=SUPPORTED_CHORUS, default="intro")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_music(
        prompt=args.prompt,
        duration_seconds=args.duration,
        seed=args.seed,
        chorus=args.chorus,
    )
    print(result.status_text)
    print(f"WAV: {result.wav_path}")
    print(f"MP3: {result.mp3_path}")


if __name__ == "__main__":
    main()
