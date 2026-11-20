"""Ścieżki runów: saved_models/pretrained/from_scratch | finetuned."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

_RUN_SEP = "___"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sanitize_tag(tag: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in tag.strip()).strip("-")


def run_dir_name(run_tag: str | None = None, when: datetime | None = None) -> str:
    from _utils_.log_print import inverted_unix_prefix

    when = when or datetime.now()
    prefix = f"{inverted_unix_prefix(when):010d}"
    date = when.strftime("%Y-%m-%d_%H-%M")
    if run_tag and run_tag.strip():
        return f"{prefix}{_RUN_SEP}{_sanitize_tag(run_tag)}{_RUN_SEP}{date}"
    return f"{prefix}{_RUN_SEP}{date}"


def phase_root(phase: str, architecture: str) -> Path:
    return project_root() / "saved_models" / phase / architecture


def resolve_evaluation_dir(run_tag: str | None = None, when: datetime | None = None) -> Path:
    root = project_root() / "model_evaluations" / run_dir_name(run_tag, when)
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_save_dir(
    architecture: str,
    *,
    phase: str = "finetuned",
    run_tag: str | None = None,
    when: datetime | None = None,
) -> Path:
    env = os.environ.get("SEG_SAVE_DIR", "").strip()
    if env:
        root = Path(env)
    else:
        root = phase_root(phase, architecture) / run_dir_name(run_tag, when)
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "plots").mkdir(exist_ok=True)
    return root


def find_latest_run_dir(architecture: str, phase: str = "finetuned") -> Path | None:
    base = phase_root(phase, architecture)
    if not base.is_dir():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name, reverse=True)[0]


def resolve_resume_from_env_or_arg() -> Path | None:
    raw = os.environ.get("SEG_RESUME", "").strip()
    if not raw:
        return None
    if raw == "latest":
        arch = os.environ.get("SEG_ARCHITECTURE", "")
        phase = os.environ.get("SEG_PHASE", "finetuned")
        if not arch:
            return None
        return find_latest_run_dir(arch, phase)
    p = Path(raw)
    return p if p.is_dir() else None


def write_run_meta(save_dir: Path, meta: dict) -> None:
    (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")