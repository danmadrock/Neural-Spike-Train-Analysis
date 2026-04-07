"""PCA wrapper used across training and inference."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class NeuralPCA:
    """Wrapper around ``sklearn.decomposition.PCA`` with project-specific guards."""

    def __init__(self, n_components: int) -> None:
        if n_components <= 0:
            msg = "n_components must be > 0"
            raise ValueError(msg)
        self.n_components = int(n_components)
        self._pca = PCA(n_components=self.n_components)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        self._ensure_fitted()
        return self._pca.explained_variance_ratio_

    def fit(self, train_matrix: np.ndarray) -> NeuralPCA:
        """Fit PCA exactly once on training data only."""
        if self._is_fitted:
            msg = "NeuralPCA.fit() may only be called once per instance"
            raise RuntimeError(msg)

        matrix = np.asarray(train_matrix, dtype=float)
        if matrix.ndim != 2:
            msg = "train_matrix must be 2D with shape (T, N)"
            raise ValueError(msg)

        self._pca.fit(matrix)
        self._is_fitted = True
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Transform any split (train/test/live) with already-fitted PCA."""
        self._ensure_fitted()
        data = np.asarray(matrix, dtype=float)
        if data.ndim != 2:
            msg = "matrix must be 2D with shape (T, N)"
            raise ValueError(msg)
        return self._pca.transform(data)

    def n_components_for_threshold(self, threshold: float) -> int:
        self._ensure_fitted()
        if not 0 < threshold <= 1:
            msg = "threshold must be in (0, 1]"
            raise ValueError(msg)
        cumulative = np.cumsum(self._pca.explained_variance_ratio_)
        return int(np.searchsorted(cumulative, threshold, side="left") + 1)

    def plot_scree(self, path: str | Path) -> Path:
        """Plot cumulative explained variance and save as PNG."""
        self._ensure_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        cumulative = np.cumsum(self._pca.explained_variance_ratio_)
        components = np.arange(1, cumulative.size + 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(components, cumulative, marker="o")
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Cumulative variance explained")
        ax.set_ylim(0, 1.01)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(target, dpi=150)
        plt.close(fig)
        return target

    def save(self, path: str | Path) -> Path:
        """Serialize the full ``NeuralPCA`` object using pickle."""
        self._ensure_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(self, f)
        return target

    @classmethod
    def load(cls, path: str | Path) -> NeuralPCA:
        source = Path(path)
        with source.open("rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, cls):
            msg = f"Pickle at {source} does not contain a NeuralPCA instance"
            raise TypeError(msg)
        return loaded

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            msg = "NeuralPCA has not been fitted yet"
            raise RuntimeError(msg)
