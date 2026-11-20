"""Orkiestracja: maski, trening, ewaluacja, porównanie — szczegóły poza main.py."""

from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from _utils_.log_print import log_print, reset_log_timer, tensorboard_run_dir
from _utils_.run_paths import project_root, resolve_evaluation_dir
from data.prepare import prepare_masks
from infer import compare_models, evaluate_run
from metrics.plot_training import plot_training_run
from metrics.plot_training_compare import main as plot_training_compare

EVAL_KEYS = ("eval_val_images", "eval_test_images", "render_images")
PREPARE_WORKERS = 8
COMPARE_EXAMPLES = 6


@dataclass
class RunOutcome:
    architecture: str
    run_dir: Path
    reports: dict[str, dict]
    failed: bool = False


def init_session(when: datetime | None = None) -> str:
    """TensorBoard root + banner GPU (jak lab03 main)."""
    reset_log_timer()
    when = when or datetime.now()
    tb = tensorboard_run_dir(
        os.environ.get("TENSORBOARD_LOGDIR", str(project_root() / "runs")),
        when,
    )
    if not torch.cuda.is_available():
        log_print("UWAGA: CUDA niedostępne — trening na CPU")
    else:
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        log_print(f"GPU: {len(names)}x {names}")
    log_print(f"TensorBoard logi: {tb}")
    return tb


def _mask_limits(presets: list[dict]) -> dict[str, int]:
    def need(*keys: str) -> int:
        vals = [int(p[k]) for p in presets for k in keys]
        return 0 if any(v == 0 for v in vals) else max(vals)

    return {
        "train": need("max_train_images"),
        "val": need("max_val_images", "eval_val_images"),
        "test": need("eval_test_images"),
    }


def ensure_masks(presets: list[dict]) -> None:
    limits = _mask_limits(presets)
    log_print(f"Maski: limity per split {limits} (0 = cały split)")
    agg = prepare_masks(limits, workers=PREPARE_WORKERS)
    log_print(f"Maski gotowe: nowe={agg['processed']} istniejące={agg['skipped']}")


def train_and_eval(
    tensorboard_dir: str,
    *,
    group: str,
    preset: dict,
    run_fn: Callable[..., Path],
    eval_root: Path,
) -> RunOutcome:
    preset = dict(preset)
    eval_cfg = {k: preset.pop(k) for k in EVAL_KEYS}
    arch = preset["architecture"]
    label = f"{group}/{arch}"
    log_print(f"=== {label} ({preset['run_tag']}) ===")
    try:
        run_dir = run_fn(tensorboard_dir=tensorboard_dir, **preset)
        plot_training_run(run_dir)
        out = eval_root / group / arch
        reports = {
            split: evaluate_run(
                run_dir,
                split=split,
                max_images=eval_cfg[f"eval_{split}_images"],
                render_images=eval_cfg["render_images"],
                out_dir=out,
            )
            for split in ("val", "test")
        }
        for split, r in reports.items():
            log_print(
                f"{label} {split}: mIoU={r['mIoU']:.4f} mIoU_fg={r['mIoU_fg']:.4f} "
                f"F1={r['det_f1']:.4f} ({r['n_images']} obr., {r['img_per_s']:.1f} img/s)"
            )
        return RunOutcome(architecture=arch, run_dir=run_dir, reports=reports)
    except Exception:
        log_print(f"BŁĄD {label}:\n{traceback.format_exc()}")
        return RunOutcome(architecture=arch, run_dir=Path(), reports={}, failed=True)


def compare_group(eval_root: Path, group: str, outcomes: list[RunOutcome]) -> None:
    ok = [o for o in outcomes if not o.failed]
    if len(ok) < 2:
        return
    entries = [(o.architecture, o.run_dir) for o in ok]
    for split in ("val", "test"):
        compare_models(
            entries,
            [o.reports[split] for o in ok],
            split=split,
            n_examples=COMPARE_EXAMPLES,
            out_dir=eval_root / group / "compare",
        )
    plot_training_compare(
        [o.run_dir for o in ok],
        eval_root / group / "compare" / "training_mIoU_fg.png",
    )


def finalize(
    eval_root: Path,
    *,
    started_at: float,
    groups: dict[str, list[RunOutcome]],
) -> None:
    failed = [f"{g}/{o.architecture}" for g, items in groups.items() for o in items if o.failed]
    summary = {
        "elapsed_min": round((time.monotonic() - started_at) / 60.0, 2),
        "failed": failed,
        "runs": {
            g: [{"architecture": o.architecture, "run_dir": str(o.run_dir)} for o in items if not o.failed]
            for g, items in groups.items()
        },
    }
    (eval_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log_print(f"Koniec: {summary['elapsed_min']} min, błędy: {failed or 'brak'} -> {eval_root}")
    if failed:
        raise SystemExit(1)