import numpy as np

from src.training.metrics import pearson_r, r2_score, rmse, velocity_rmse


def test_r2_score_known_answer():
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[1.0], [2.0], [4.0]])
    assert np.isclose(r2_score(y_true, y_pred), 0.5)


def test_r2_score_constant_target_edge_cases():
    y_true = np.ones((5, 2))
    assert r2_score(y_true, y_true) == 1.0
    assert r2_score(y_true, np.zeros_like(y_true)) == 0.0


def test_rmse_known_answer():
    y_true = np.array([[0.0, 0.0], [2.0, 2.0]])
    y_pred = np.array([[0.0, 0.0], [0.0, 0.0]])
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(2.0))


def test_pearson_r_axiswise_mean():
    y_true = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 5.0]])
    y_pred = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])
    assert np.isclose(pearson_r(y_true, y_pred), 1.0)


def test_velocity_rmse_known_answer():
    y_true = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 2.0]])
    y_pred = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    # true velocities: [[1,1], [2,1]]; pred velocities: [[0,1], [2,0]]
    # squared errors: [[1,0], [0,1]] mean=0.5 => sqrt=0.7071
    assert np.isclose(velocity_rmse(y_true, y_pred), np.sqrt(0.5))
