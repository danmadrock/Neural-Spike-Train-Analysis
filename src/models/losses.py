"""Composite trajectory loss."""

from __future__ import annotations

import torch


def trajectory_loss(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    model: torch.nn.Module,
    velocity_lambda: float = 0.1,
    l1_lambda: float = 1e-5,
) -> torch.Tensor:
    """Compute composite trajectory loss used by sequence decoders.

    The objective combines three terms:
    1. **Heteroscedastic Gaussian NLL** for calibrated mean/variance prediction.
    2. **Velocity MSE penalty** on first differences to enforce smooth dynamics.
    3. **Explicit L1 norm penalty** over all trainable parameters for sparsity.

    L2 regularization is intentionally excluded here and should be applied through
    AdamW ``weight_decay``.
    """

    if pred_mean.shape != target.shape or pred_logvar.shape != target.shape:
        raise ValueError("pred_mean, pred_logvar, and target must have same shape")

    inv_var = torch.exp(-pred_logvar)
    nll = 0.5 * (pred_logvar + (target - pred_mean).pow(2) * inv_var)
    nll_term = nll.mean()

    pred_vel = pred_mean[:, 1:, :] - pred_mean[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    velocity_term = torch.mean((pred_vel - target_vel).pow(2))

    l1_term = torch.zeros((), device=pred_mean.device)
    for p in model.parameters():
        l1_term = l1_term + p.abs().sum()

    return nll_term + velocity_lambda * velocity_term + l1_lambda * l1_term
