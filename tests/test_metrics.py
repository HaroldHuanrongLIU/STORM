from __future__ import annotations

import pytest
import torch

from storm_surgwmbench.evaluation.metrics import (
    ade,
    discrete_frechet,
    endpoint_error,
    error_by_horizon,
    fde,
    symmetric_hausdorff,
    trajectory_length,
    trajectory_length_error,
    trajectory_smoothness,
)


def test_metrics_simple_trajectory_values() -> None:
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pred = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

    assert ade(pred, target) == pytest.approx(1.0 / 3.0)
    assert fde(pred, target) == pytest.approx(0.0)
    assert endpoint_error(pred, target) == pytest.approx(0.0)
    assert discrete_frechet(pred, target) == pytest.approx(1.0)
    assert symmetric_hausdorff(pred, target) == pytest.approx(1.0)
    assert trajectory_length(target) == pytest.approx(2.0)
    assert trajectory_length_error(pred, target) == pytest.approx(2.0**0.5 * 2.0 - 2.0)
    assert trajectory_smoothness(target) == pytest.approx(0.0)
    assert error_by_horizon(pred, target, [1, 2, 3]) == {1: 0.0, 2: 1.0, 3: 0.0}


def test_metrics_support_batch_masks_and_invalid_masks() -> None:
    target = torch.tensor([[[0.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]])
    pred = target + 1.0
    mask = torch.tensor([[True, False], [True, True]])

    assert ade(pred, target, mask) == pytest.approx(2.0**0.5)
    assert fde(pred, target, mask) == pytest.approx(2.0**0.5)
    assert ade(pred, target, torch.zeros_like(mask, dtype=torch.bool)) is None
