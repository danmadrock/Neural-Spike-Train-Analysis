"""MC Dropout inference utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class UncertaintyResult:
    """Container with MC Dropout aggregates and sampled trajectories."""

    mean: np.ndarray
    std: np.ndarray
    samples: np.ndarray


def mc_predict(
    model: nn.Module, latents: torch.Tensor, n_samples: int = 50
) -> UncertaintyResult:
    """Run MC Dropout inference and return sample statistics.

    Args:
        model: Sequence decoder returning ``(mean, log_var)``.
        latents: Input tensor of shape ``(batch, T, K)``.
        n_samples: Number of stochastic forward passes.

    Returns:
        ``UncertaintyResult`` with arrays shaped ``(batch, T, 2)`` for ``mean`` and
        ``std`` and ``(n_samples, batch, T, 2)`` for ``samples``.
    """

    if n_samples < 1:
        msg = "n_samples must be >= 1"
        raise ValueError(msg)

    was_training = model.training
    model.train()

    with torch.no_grad():
        draws: list[np.ndarray] = []
        for _ in range(n_samples):
            pred_mean, _ = model(latents)
            draws.append(pred_mean.detach().cpu().numpy())

    if not was_training:
        model.eval()

    samples = np.stack(draws, axis=0)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)

    return UncertaintyResult(mean=mean, std=std, samples=samples)
