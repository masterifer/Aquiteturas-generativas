"""Funcoes auxiliares compartilhadas."""

from __future__ import annotations

import random
from typing import Iterable

import torch


try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def set_seed(seed: int) -> None:
    """Define a seed para tornar os experimentos mais reproduziveis."""
    random.seed(seed)
    torch.manual_seed(seed)

    if np is not None:
        np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Conta parametros treinaveis do modelo."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Retorna CUDA se disponivel e solicitado, senao CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reset_cuda_peak_memory(device: torch.device) -> None:
    """Reseta o medidor de memoria de pico em CUDA."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_cuda_peak_memory_mb(device: torch.device) -> float | None:
    """Retorna memoria de pico em MB quando usando CUDA."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
