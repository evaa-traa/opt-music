# Opt Music

Colab-first open-source instrumental background music generator for YouTube Shorts, built around `FunAudioLLM/InspireMusic-Base-24kHz`.

This repo keeps the real application code in GitHub so your Colab notebook stays short. Colab only needs to clone the repo, run the bootstrap script, and launch the app.

## What changed

- Switched from `facebook/musicgen-small` to `FunAudioLLM/InspireMusic-Base-24kHz`
- Switched from Hugging Face Spaces CPU focus to a Colab-first workflow
- Removed loop-extension generation
- Limited generation to true end-to-end `15s` and `30s` clips, which matches the Base-24kHz model family

## Features

- Prompt-based instrumental music generation
- Presets: `Epic History`, `Phonk`, `Trailer`, `Documentary`
- Seed control
- Section bias control: `intro`, `verse`, `chorus`, `outro`
- WAV preview
- WAV and MP3 download
- Loudness normalization to `-18 LUFS`
- GitHub-hosted code plus thin Colab launcher flow

## Project structure

- `app.py`: Gradio UI
- `model.py`: InspireMusic bootstrap-aware wrapper and generation logic
- `utils.py`: audio loading, loudness normalization, WAV/MP3 export
- `bootstrap.py`: clones InspireMusic, installs dependencies, downloads model weights
- `generate.py`: CLI for one-shot generation
- `requirements.txt`: lightweight top-level dependencies
- `README.md`: setup guide

## Colab quick start

Use a GPU runtime if possible.

```python
!git clone https://github.com/<your-user>/<your-repo>.git
%cd <your-repo>
!python bootstrap.py
!python app.py --share
```

That is the intended workflow. The heavy code lives in this repo and the official InspireMusic repo is pulled automatically by `bootstrap.py`.

## One-shot Colab generation

```python
!git clone https://github.com/<your-user>/<your-repo>.git
%cd <your-repo>
!python bootstrap.py
!python generate.py --prompt "epic historical instrumental soundtrack, no vocals" --duration 30 --seed 42 --chorus intro
```

## Local setup

1. Clone this repo.
2. Install the lightweight dependencies:

```bash
pip install -r requirements.txt
```

3. Bootstrap the full InspireMusic runtime:

```bash
python bootstrap.py
```

4. Launch the app:

```bash
python app.py
```

## Model notes

- Default model: `FunAudioLLM/InspireMusic-Base-24kHz`
- Model repo: https://huggingface.co/FunAudioLLM/InspireMusic-Base-24kHz
- Upstream code repo: https://github.com/FunAudioLLM/InspireMusic
- This repo uses the official InspireMusic codebase as a vendor dependency pulled at bootstrap time

## Duration limits

This repo currently supports only:

- `15 seconds`
- `30 seconds`

That is intentional. The Base-24kHz model is the practical target here. If you want true long-form generation later, you should evaluate `InspireMusic-1.5B-Long` separately.

## Output

Each generation produces:

- normalized WAV
- normalized MP3
- playback in the UI

Files are written to a temporary output directory during runtime.

## Why the bootstrap script exists

The upstream InspireMusic project is too large and dependency-heavy to recreate manually in a notebook every time. `bootstrap.py` reduces Colab to a few commands by:

- installing system packages for Colab when needed
- cloning the official InspireMusic repository with submodules
- installing Python dependencies
- downloading the requested model snapshot from Hugging Face
- patching the downloaded model YAML paths for the local runtime layout

## Practical expectations

- First run is slow because the upstream code and model weights must be fetched
- GPU Colab runtimes are the intended target
- CPU may still work, but expect long runtimes and possible memory pressure
- This repo does not currently target Hugging Face Spaces free CPU

## License and due diligence

Before using any generated music commercially, review the current license and model card for the upstream model yourself:

- https://huggingface.co/FunAudioLLM/InspireMusic-Base-24kHz
- https://huggingface.co/FunAudioLLM/InspireMusic-Base-24kHz/blob/main/LICENSE.txt

This repo is a wrapper around that upstream model and codebase. You are responsible for checking whether the upstream license and your intended use are compatible.
