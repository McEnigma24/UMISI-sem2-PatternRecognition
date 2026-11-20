"""Augmentacje treningowe na GPU (uint8/float batch już na urządzeniu)."""

from __future__ import annotations

import torch


def gpu_train_augment(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """image float [B,3,H,W] w [0,1]; mask long [B,H,W]. Zwraca (image, mask)."""
    b = image.shape[0]
    if b == 0:
        return image, mask
    device = image.device

    if torch.rand(1, device=device).item() < 0.5:
        image = torch.flip(image, dims=(3,))
        mask = torch.flip(mask, dims=(2,))

    for i in range(b):
        if torch.rand(1, device=device).item() < 0.3:
            factor = 0.85 + 0.3 * torch.rand(1, device=device).item()
            image[i] = (image[i] * factor).clamp(0.0, 1.0)

        if torch.rand(1, device=device).item() < 0.3:
            gray = image[i].mean(dim=0, keepdim=True)
            sat = 0.7 + 0.6 * torch.rand(1, device=device).item()
            image[i] = (sat * image[i] + (1.0 - sat) * gray).clamp(0.0, 1.0)

        if torch.rand(1, device=device).item() < 0.15:
            noise = torch.randn_like(image[i]) * 0.02
            image[i] = (image[i] + noise).clamp(0.0, 1.0)

    return image, mask