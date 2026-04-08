import torch

from src.models.losses import trajectory_loss
from src.models.lstm_decoder import LSTMDecoder


def test_trajectory_loss_returns_scalar():
    model = LSTMDecoder(input_size=10)
    x = torch.randn(3, 20, 10)
    target = torch.randn(3, 20, 2)
    mean, log_var = model(x)

    loss = trajectory_loss(mean, log_var, target, model)
    assert loss.ndim == 0


def test_gradients_flow_through_both_heads():
    model = LSTMDecoder(input_size=10)
    x = torch.randn(3, 20, 10)
    target = torch.randn(3, 20, 2)
    mean, log_var = model(x)

    loss = trajectory_loss(mean, log_var, target, model)
    loss.backward()

    assert model.mean_head.weight.grad is not None
    assert model.log_var_head.weight.grad is not None
    assert torch.any(model.mean_head.weight.grad != 0)
    assert torch.any(model.log_var_head.weight.grad != 0)
