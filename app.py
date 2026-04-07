from __future__ import annotations

import argparse
import os
import sys

import gradio as gr

from model import PRESET_PROMPTS, RuntimeSetupError, SUPPORTED_CHORUS, SUPPORTED_DURATIONS, generate_music


TITLE = "Opt Music"
DESCRIPTION = """
Generate instrumental YouTube Shorts background music with `FunAudioLLM/InspireMusic-Base-24kHz`.

- Colab-first workflow with GitHub-hosted code
- True end-to-end generation for 15s and 30s clips
- Presets tuned for Shorts-friendly instrumental beds
- WAV preview plus WAV and MP3 downloads
"""


def is_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ


def use_preset(name: str) -> str:
    return PRESET_PROMPTS[name]


def run_generation(prompt: str, duration: int, seed: int, chorus: str):
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        raise gr.Error("Enter a prompt or click a preset.")

    try:
        result = generate_music(
            prompt=cleaned_prompt,
            duration_seconds=int(duration),
            seed=int(seed),
            chorus=chorus,
        )
    except RuntimeSetupError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}") from exc

    return result.wav_path, result.wav_path, result.mp3_path, result.status_text


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    gr.Markdown(
        "Run `python bootstrap.py` before launching the app. "
        "The Base-24kHz model supports true 15s and 30s generations."
    )

    with gr.Row():
        epic_button = gr.Button("Epic History")
        phonk_button = gr.Button("Phonk")
        trailer_button = gr.Button("Trailer")
        documentary_button = gr.Button("Documentary")

    prompt_box = gr.Textbox(
        label="Prompt",
        lines=3,
        value=PRESET_PROMPTS["Epic History"],
        placeholder="Describe the instrumental track you want.",
    )

    with gr.Row():
        duration_input = gr.Dropdown(
            label="Duration (seconds)",
            choices=[str(item) for item in SUPPORTED_DURATIONS],
            value=str(SUPPORTED_DURATIONS[-1]),
        )
        chorus_input = gr.Dropdown(
            label="Section shape",
            choices=list(SUPPORTED_CHORUS),
            value="intro",
            info="Controls the section bias exposed by InspireMusic.",
        )
        seed_input = gr.Number(label="Seed", value=42, precision=0)

    generate_button = gr.Button("Generate Music", variant="primary")

    preview_audio = gr.Audio(label="Playback", type="filepath")
    with gr.Row():
        wav_file = gr.File(label="Download WAV")
        mp3_file = gr.File(label="Download MP3")

    status_box = gr.Markdown("Ready.")

    gr.Examples(
        examples=[
            [PRESET_PROMPTS["Epic History"], "30", 42, "intro"],
            [PRESET_PROMPTS["Phonk"], "15", 7, "verse"],
            [PRESET_PROMPTS["Trailer"], "30", 123, "chorus"],
            [PRESET_PROMPTS["Documentary"], "15", 99, "outro"],
        ],
        inputs=[prompt_box, duration_input, seed_input, chorus_input],
    )

    epic_button.click(fn=lambda: use_preset("Epic History"), outputs=prompt_box)
    phonk_button.click(fn=lambda: use_preset("Phonk"), outputs=prompt_box)
    trailer_button.click(fn=lambda: use_preset("Trailer"), outputs=prompt_box)
    documentary_button.click(fn=lambda: use_preset("Documentary"), outputs=prompt_box)

    generate_button.click(
        fn=run_generation,
        inputs=[prompt_box, duration_input, seed_input, chorus_input],
        outputs=[preview_audio, wav_file, mp3_file, status_box],
        api_name="generate",
    )

demo.queue(max_size=4, default_concurrency_limit=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Opt Music Gradio app.")
    parser.add_argument("--share", action="store_true", help="Enable a public Gradio share link.")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=int(os.getenv("PORT", "7860")))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share or is_colab(),
    )
