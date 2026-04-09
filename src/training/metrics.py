"""Training and evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch


def _as_2d_float(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=float)
    if data.ndim == 1:
        return data[:, None]
    if data.ndim == 2:
        return data
    msg = "Expected 1D or 2D array for metric computation"
    raise ValueError(msg)


def _validate_shapes(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_true_2d = _as_2d_float(y_true)
    y_pred_2d = _as_2d_float(y_pred)
    if y_true_2d.shape != y_pred_2d.shape:
        msg = f"Shape mismatch: y_true{y_true_2d.shape} != y_pred{y_pred_2d.shape}"
        raise ValueError(msg)
    return y_true_2d, y_pred_2d


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Variance-explained score with sklearn-like constant-target handling."""
    yt, yp = _validate_shapes(y_true, y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    centered = yt - np.mean(yt, axis=0, keepdims=True)
    ss_tot = float(np.sum(centered**2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _validate_shapes(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Pearson correlation across axes, robust to constant vectors."""
    yt, yp = _validate_shapes(y_true, y_pred)
    per_axis: list[float] = []
    for axis in range(yt.shape[1]):
        t = yt[:, axis]
        p = yp[:, axis]
        t_std = float(np.std(t))
        p_std = float(np.std(p))
        if t_std <= 1e-12 or p_std <= 1e-12:
            per_axis.append(0.0)
        else:
            per_axis.append(float(np.corrcoef(t, p)[0, 1]))
    return float(np.mean(per_axis))


def velocity_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _validate_shapes(y_true, y_pred)
    if yt.shape[0] < 2:
        return 0.0
    vel_true = np.diff(yt, axis=0)
    vel_pred = np.diff(yp, axis=0)
    return float(np.sqrt(np.mean((vel_true - vel_pred) ** 2)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return rmse(y_true, y_pred)


def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    mean_true = torch.mean(y_true, dim=(0, 1), keepdim=True)
    ss_tot = torch.sum((y_true - mean_true) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def rmse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
