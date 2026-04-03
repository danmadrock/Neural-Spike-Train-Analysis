def test_synthetic_spike_trains_shape(synthetic_spike_trains):
    assert len(synthetic_spike_trains) == 3
    for train in synthetic_spike_trains:
        assert train.ndim == 1
        assert (train >= 0).all()
        assert (train <= 2.0).all()


def test_synthetic_trial_shapes(synthetic_trial):
    T, K = synthetic_trial["T"], synthetic_trial["K"]
    assert synthetic_trial["latents"].shape == (T, K)
    assert synthetic_trial["trajectory"].shape == (T, 2)
