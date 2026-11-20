#!/usr/bin/env python3
"""Weryfikacja PyTorch / CUDA — uruchom: ./scripts/drun verify"""

from __future__ import annotations

import os
import sys


def main() -> int:
    import torch

    print("=== PyTorch / CUDA ===")
    print(f"PyTorch:  {torch.__version__}")
    print(f"CUDA:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:    {torch.cuda.get_device_name(i)}")
        print(f"CUDA ver: {torch.version.cuda}")
        print(f"bf16:     {torch.cuda.is_bf16_supported()}")
        print("\n=== Gotowe ===")
        return 0

    if os.environ.get("ML_TEMPLATE_GPU") == "1":
        print("BŁĄD: wymuszono GPU (ML_TEMPLATE_GPU=1), ale CUDA niedostępne.")
        return 1

    print("CUDA niedostępne — kontener działa na CPU (OK jeśli ML_TEMPLATE_GPU nie wymusza GPU).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
