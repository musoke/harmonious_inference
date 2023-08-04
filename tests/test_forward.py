from typing import Type

import numpy as np
import pytest
import scipy

from harmonious_inference import BasePhysicsModel, PhysicsModel, PhysicsModelDetailed

dist = scipy.stats.distributions.uniform(-1, 1)


def test_cannot_init_base() -> None:
    """Abstract class cannot be instantiated."""
    with pytest.raises(TypeError):
        BasePhysicsModel(dist, dist, np.arange(0, 1), 1.0)  # type: ignore


@pytest.mark.parametrize("model", [PhysicsModel, PhysicsModelDetailed])
def test_can_init(model: Type[BasePhysicsModel]) -> None:
    m = model(dist, dist, np.arange(0, 1), 1.0)

    p = np.array([1, 2, 2, 4, 4])

    assert isinstance(m.forward_model(p), np.ndarray)
