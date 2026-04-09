import torch

from src.models.gru_decoder import GRUDecoder
from src.models.lstm_decoder import LSTMDecoder


def test_lstm_forward_shapes():
    model = LSTMDecoder(input_size=15, hidden_size=256, num_layers=2, dropout=0.3)
    x = torch.randn(4, 40, 15)
    mean, log_var = model(x)
    assert mean.shape == (4, 40, 2)
    assert log_var.shape == (4, 40, 2)


def test_gru_forward_shapes():
    model = GRUDecoder(input_size=15, hidden_size=256, num_layers=2, dropout=0.3)
    x = torch.randn(4, 40, 15)
    mean, log_var = model(x)
    assert mean.shape == (4, 40, 2)
    assert log_var.shape == (4, 40, 2)


def test_decoders_share_interface():
    x = torch.randn(2, 10, 8)
    for decoder_cls in (LSTMDecoder, GRUDecoder):
        model = decoder_cls(input_size=8)
        out = model(x)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == out[1].shape == (2, 10, 2)
