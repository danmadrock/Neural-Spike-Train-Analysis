"""Linear Wiener filter baseline decoder"""

from __future__ import annotations

import mlflow
import numpy as np
from scipy.linalg import lstsq

from src.training.metrics import r2_score_np


class WienerFilter:
    """Least-squares linear decoder from neural rates to 2D trajectory"""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, rates: np.ndarray, trajectory: np.ndarray) -> WienerFilter:
        x = np.asarray(rates, dtype=float)
        y = np.asarray(trajectory, dtype=float)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("rates and trajectory must be 2D")
        if x.shape[0] != y.shape[0]:
            raise ValueError("rates and trajectory must share time dimension")

        x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        solution = lstsq(x_aug, y)
        if solution is None:
            raise RuntimeError("scipy.linalg.lstsq returned no solution")
        self.coef_ = np.asarray(solution[0], dtype=float)
        return self

    def predict(self, rates: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("WienerFilter is not fitted")
        x = np.asarray(rates, dtype=float)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        return x_aug @ self.coef_


def evaluate_wiener_and_log(
    train_rates: np.ndarray,
    train_traj: np.ndarray,
    val_rates: np.ndarray,
    val_traj: np.ndarray,
) -> tuple[WienerFilter, float]:
    """Fit Wiener baseline, compute validation R², and log to MLflow."""

    model = WienerFilter().fit(train_rates, train_traj)
    pred = model.predict(val_rates)
    r2 = r2_score_np(val_traj, pred)
    mlflow.log_metric("wiener_val_r2", float(r2))
    return model, float(r2)
