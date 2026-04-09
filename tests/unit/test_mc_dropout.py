import numpy as np
import torch

from src.models.lstm_decoder import LSTMDecoder
from src.uncertainty.mc_dropout import UncertaintyResult, mc_predict


def test_mc_predict_returns_expected_shapes_and_nonzero_std():
    torch.manual_seed(7)
    model = LSTMDecoder(input_size=8, hidden_size=32, num_layers=2, dropout=0.4)
    latents = torch.randn(3, 16, 8)

    result = mc_predict(model, latents, n_samples=30)

    assert isinstance(result, UncertaintyResult)
    assert result.mean.shape == (3, 16, 2)
    assert result.std.shape == (3, 16, 2)
    assert result.samples.shape == (30, 3, 16, 2)
    assert np.all(result.std > 0)


def test_uncertainty_result_fields_are_exposed():
    samples = np.ones((2, 1, 3, 2), dtype=float)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)

    result = UncertaintyResult(mean=mean, std=std, samples=samples)

    assert np.array_equal(result.mean, mean)
    assert np.array_equal(result.std, std)
    assert np.array_equal(result.samples, samples)
