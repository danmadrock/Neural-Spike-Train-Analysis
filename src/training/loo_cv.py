"""Leave-one-trial-out split utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypedDict

import numpy as np

from src.reduction import NeuralPCA
from src.training.metrics import pearson_r, r2_score, rmse, velocity_rmse


class FoldScores(TypedDict):
    r2: float
    rmse: float
    pearson_r: float
    velocity_rmse: float


class LooCvScores(TypedDict):
    r2_mean: float
    r2_std: float
    rmse_mean: float
    rmse_std: float
    pearson_r_mean: float
    pearson_r_std: float
    velocity_rmse_mean: float
    velocity_rmse_std: float
    folds: list[FoldScores]


def leave_one_trial_out(trials: Sequence[object]) -> Iterator[tuple[list[int], int]]:
    """Yield ``(train_indices, test_index)`` for every trial.
    The test index is always excluded from train indices, which keeps fold-level
    preprocessing (e.g., PCA.fit) restricted to training data only.
    """
    n_trials = len(trials)
    if n_trials == 0:
        return

    all_indices = list(range(n_trials))
    for test_index in all_indices:
        train_indices = [idx for idx in all_indices if idx != test_index]
        yield train_indices, test_index


def loo_cv(
    neural_trials: Sequence[np.ndarray],
    trajectory_trials: Sequence[np.ndarray],
    *,
    pca_components: int,
    model_factory: Callable[[], Any],
    fit_fn: Callable[[Any, np.ndarray, np.ndarray], None],
    predict_fn: Callable[[Any, np.ndarray], np.ndarray],
) -> LooCvScores:
    """Run fold-wise LOO-CV with per-fold PCA fitting and held-out evaluation."""
    if len(neural_trials) != len(trajectory_trials):
        msg = "neural_trials and trajectory_trials must have identical lengths"
        raise ValueError(msg)
    if len(neural_trials) < 2:
        msg = "LOO-CV requires at least two trials"
        raise ValueError(msg)

    folds: list[FoldScores] = []
    for train_indices, test_index in leave_one_trial_out(neural_trials):
        train_rates = [np.asarray(neural_trials[i], dtype=float) for i in train_indices]
        test_rates = np.asarray(neural_trials[test_index], dtype=float)
        train_targets = [
            np.asarray(trajectory_trials[i], dtype=float) for i in train_indices
        ]
        test_target = np.asarray(trajectory_trials[test_index], dtype=float)

        pca = NeuralPCA(n_components=pca_components).fit(np.vstack(train_rates))
        train_latents = np.vstack([pca.transform(trial) for trial in train_rates])
        test_latents = pca.transform(test_rates)

        train_target_stacked = np.vstack(train_targets)
        model = model_factory()
        fit_fn(model, train_latents, train_target_stacked)
        test_pred = np.asarray(predict_fn(model, test_latents), dtype=float)

        fold_scores: FoldScores = {
            "r2": r2_score(test_target, test_pred),
            "rmse": rmse(test_target, test_pred),
            "pearson_r": pearson_r(test_target, test_pred),
            "velocity_rmse": velocity_rmse(test_target, test_pred),
        }
        folds.append(fold_scores)

    r2_values = np.asarray([fold["r2"] for fold in folds], dtype=float)
    rmse_values = np.asarray([fold["rmse"] for fold in folds], dtype=float)
    corr_values = np.asarray([fold["pearson_r"] for fold in folds], dtype=float)
    vel_values = np.asarray([fold["velocity_rmse"] for fold in folds], dtype=float)
    return {
        "r2_mean": float(np.mean(r2_values)),
        "r2_std": float(np.std(r2_values, ddof=0)),
        "rmse_mean": float(np.mean(rmse_values)),
        "rmse_std": float(np.std(rmse_values, ddof=0)),
        "pearson_r_mean": float(np.mean(corr_values)),
        "pearson_r_std": float(np.std(corr_values, ddof=0)),
        "velocity_rmse_mean": float(np.mean(vel_values)),
        "velocity_rmse_std": float(np.std(vel_values, ddof=0)),
        "folds": folds,
    }
