"""Wykresy pojedynczego runu segmentacji (metrics.csv) -> <run>/plots/."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from classes import CLASS_NAMES


def plot_training_run(save_dir: Path, *, dpi: int = 120) -> list[Path]:
    """Lightning: loss, mIoU, IoU per klasa. YOLO: ultralytics zapisuje własne results.png."""
    save_dir = Path(save_dir)
    csv_path = save_dir / "metrics.csv"
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    arch = save_dir.parent.name
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _plot(cols: list[str], ylabel: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in cols:
            if c in df.columns:
                ax.plot(df["step"], df[c], marker="o", label=c)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — {arch}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = plots_dir / fname
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        saved.append(out)

    _plot(["train_loss", "val_loss"], "loss", "loss.png")
    _plot(["val_mIoU", "val_mIoU_fg"], "mIoU", "miou.png")
    _plot([f"iou_{n}" for n in CLASS_NAMES], "IoU per klasa (val)", "iou_per_class.png")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=Path)
    args = parser.parse_args()
    for p in plot_training_run(args.save_dir):
        print(p)


if __name__ == "__main__":
    main()
