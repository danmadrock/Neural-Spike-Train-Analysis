"""Training and evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    mean_true = torch.mean(y_true, dim=(0, 1), keepdim=True)
    ss_tot = torch.sum((y_true - mean_true) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def rmse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
