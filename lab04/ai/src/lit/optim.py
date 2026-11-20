from __future__ import annotations

import math

import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def param_groups_no_decay(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def warmup_cosine_scheduler(optimizer, *, warmup_steps: int, total_steps: int) -> LambdaLR:
    total = max(total_steps, 1)
    warmup = min(warmup_steps, max(1, total // 20))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        if total <= warmup:
            return 1.0
        p = (step - warmup) / float(max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * p))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def scaled_learning_rate(
    *,
    baseline_batch_size: int,
    baseline_learning_rate: float,
    global_batch: int,
    lr_scaling: str,
    learning_rate_max: float,
) -> float:
    ratio = global_batch / float(baseline_batch_size)
    if lr_scaling == "linear":
        lr = baseline_learning_rate * ratio
    else:
        lr = baseline_learning_rate * math.sqrt(ratio)
    return min(lr, learning_rate_max)