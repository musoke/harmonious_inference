from typing import Any

import numpy as np
import pytest
import sbi
import scipy
import torch

from harmonious_inference import Inferrer, PhysicsModel, PhysicsModelDetailed

dist = scipy.stats.distributions.uniform(-1, 1)


@pytest.mark.parametrize("model_class", [PhysicsModel, PhysicsModelDetailed])
def test_train_and_infer(model_class: Any) -> None:
    model = model_class(dist, dist, [0, 1], 1.0)

    t_eval = np.linspace(0, 1)

    m_range = (1.0, 5.0)
    d_range = (0, +1e0)
    k2_range = (-1e1, +1e1)
    k3_range = (-1e-1, +1e-1)
    k4_range = (+1e-9, +1e-2)

    prior = sbi.utils.BoxUniform(
        low=torch.tensor(
            [m_range[0], d_range[0], k2_range[0], k3_range[0], k4_range[0]]
        ),
        high=torch.tensor(
            [m_range[1], d_range[1], k2_range[1], k3_range[1], k4_range[1]]
        ),
    )

    inferrer = Inferrer(t_eval, prior, model)
    inferrer.train(num_simulations=10)

    true_p = prior.sample()
    observed_data = model.forward_model(true_p)

    inferrer.sample_posterior(observed_data.flatten(), 10)
