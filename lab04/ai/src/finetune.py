"""Fine-tuning wag pobranych -> saved_models/finetuned."""

from __future__ import annotations

from pathlib import Path

from train import launch_job


def main(tensorboard_dir: str, **preset) -> Path:
    return launch_job(phase="finetuned", source="downloaded", tensorboard_dir=tensorboard_dir, preset=preset)