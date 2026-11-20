"""Maski -> fine-tuning -> (opcjonalnie od zera) -> ewaluacja.

Uruchomienie: ./normal_process_start.sh  |  pełny run w tle: ./background_process_start.sh
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime

import finetune
import train
from _utils_.run_paths import resolve_evaluation_dir
from finetune_presets import FINETUNE_ALL
from pipeline import compare_group, ensure_masks, finalize, init_session, train_and_eval
from training_presets import SCRATCH_ALL


def main(session_started_at: datetime | None = None) -> str:
    """Sesja: TensorBoard + banner GPU (wzór lab03)."""
    return init_session(session_started_at)


if __name__ == "__main__":
    when = datetime.now()
    t0 = time.monotonic()
    tensorboard_dir = main()
    subprocess.run(["clear"], check=False)

    active_presets = [*FINETUNE_ALL]
    # active_presets += SCRATCH_ALL
    if not active_presets:
        raise SystemExit("Brak aktywnych presetów")

    ensure_masks(active_presets)
    eval_root = resolve_evaluation_dir("main", when)

    finetuned = [
        train_and_eval(tensorboard_dir, group="finetuned", preset=p, run_fn=finetune.main, eval_root=eval_root)
        for p in FINETUNE_ALL
    ]
    groups: dict[str, list] = {"finetuned": finetuned}

    # from_scratch = [
    #     train_and_eval(tensorboard_dir, group="from_scratch", preset=p, run_fn=train.main, eval_root=eval_root)
    #     for p in SCRATCH_ALL
    # ]
    # groups["from_scratch"] = from_scratch

    for group, outcomes in groups.items():
        compare_group(eval_root, group, outcomes)

    finalize(eval_root, started_at=t0, groups=groups)