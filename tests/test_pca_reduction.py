import numpy as np
import pytest

from src.reduction import NeuralPCA
from src.training import leave_one_trial_out


def test_neural_pca_prevents_fit_on_test_data(monkeypatch):
    trials = [
        np.full((4, 3), fill_value=1.0),
        np.full((4, 3), fill_value=2.0),
        np.full((4, 3), fill_value=3.0),
    ]

    captured_train_rows: list[np.ndarray] = []
    original_fit = NeuralPCA.fit

    def _capturing_fit(self, train_matrix):
        captured_train_rows.append(np.asarray(train_matrix).copy())
        return original_fit(self, train_matrix)

    monkeypatch.setattr(NeuralPCA, "fit", _capturing_fit)

    for train_indices, test_index in leave_one_trial_out(trials):
        train_matrix = np.vstack([trials[i] for i in train_indices])
        test_matrix = trials[test_index]

        pca = NeuralPCA(n_components=2)
        pca.fit(train_matrix)

        # Ensure train matrix values do not include the held-out trial marker.
        assert not np.any(np.isin(captured_train_rows[-1], np.unique(test_matrix)))


def test_transform_shape_matches_expected_time_and_components():
    train = np.random.default_rng(0).normal(size=(20, 6))
    test = np.random.default_rng(1).normal(size=(8, 6))

    pca = NeuralPCA(n_components=4).fit(train)
    transformed = pca.transform(test)

    assert transformed.shape == (8, 4)


def test_save_load_round_trip(tmp_path):
    matrix = np.random.default_rng(42).normal(size=(30, 5))

    pca = NeuralPCA(n_components=3).fit(matrix)
    path = tmp_path / "neural_pca.pkl"
    pca.save(path)

    loaded = NeuralPCA.load(path)

    np.testing.assert_allclose(pca.transform(matrix), loaded.transform(matrix))


def test_fit_only_once():
    matrix = np.random.default_rng(123).normal(size=(10, 4))
    pca = NeuralPCA(n_components=2).fit(matrix)
    with pytest.raises(RuntimeError):
        pca.fit(matrix)
