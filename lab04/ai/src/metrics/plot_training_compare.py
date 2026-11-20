"""Porównanie krzywych val mIoU_fg z metrics.csv kilku runów."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(run_dirs: list[Path], out: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for rd in run_dirs:
        csv = rd / "metrics.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        ax.plot(df["step"], df["val_mIoU_fg"], marker="o", label=f"{rd.parent.name}/{rd.name}")
        plotted += 1
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("step")
    ax.set_ylabel("val mIoU (fg)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("model_evaluations/training_compare.png"))
    args = p.parse_args()
    print(main(args.dirs, args.out))
