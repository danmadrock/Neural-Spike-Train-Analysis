"""Leave-one-trial-out split utilities."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


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
