from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import getdist
import numpy as np
import scipy
import torch
from getdist import plots
from sbi.inference.base import infer
from scipy.integrate import solve_ivp


class BasePhysicsModel(ABC):
    """An abstract class that all the models should inherit from.

    It does nothing other than define a common interface.
    """

    def __init__(
        self,
        x0_dist: scipy.stats.Distribution,
        v0_dist: scipy.stats.Distribution,
        t_eval: np.ndarray,
        noise_amplitude: float,
    ):
        self.x0_dist = x0_dist
        self.v0_dist = v0_dist
        self.t_eval = t_eval
        self.noise_amplitude = noise_amplitude

    @abstractmethod
    def forward_model(self, p: np.ndarray) -> np.ndarray:
        """Run model forward subject to parameters `p`."""
        pass

    def check_solution(self, sol: Any) -> np.ndarray:
        if sol.success:
            y_noise = np.random.normal(
                loc=0.0, scale=self.noise_amplitude, size=sol.y.shape
            )
            y = sol.y + y_noise

            return y
        else:
            print("Warning: solving failed, returning nans")
            return np.full((2, self.t_eval.shape[0]), np.nan)


class PhysicsModel(BasePhysicsModel):
    """A model implementing a mass-spring system and assuming Hooke's law."""

    def forward_model(self, parameters: np.ndarray) -> np.ndarray:
        m = parameters[0]
        k = parameters[1]

        def simple_harmonic_motion(t: float, y: np.ndarray) -> np.ndarray:
            dx = y[1]
            dv = -2 * k / m * y[0]
            return np.array([dx, dv])

        t_span: Tuple[float, float] = (self.t_eval[0], self.t_eval[-1])
        y0 = np.array([self.x0_dist.rvs(), self.v0_dist.rvs()])

        sol = solve_ivp(simple_harmonic_motion, t_span, y0, t_eval=self.t_eval)

        return self.check_solution(sol)


class PhysicsModelDetailed(BasePhysicsModel):
    """A model implementing a damped anharmonic oscillator."""

    def forward_model(self, parameters: np.ndarray) -> np.ndarray:
        m = parameters[0]
        d = parameters[1]
        k = np.array(parameters[2:])

        if k.shape[0] % 2 == 0:
            print(f"Warning: possibly unstable system {k}=")
        if k[-1] <= 0:
            print(f"Warning: possibly unstable system {k}=")

        def equation_of_motion(t: float, y: np.ndarray) -> np.ndarray:
            dx = y[1]
            dv = (
                -d / m * y[1]
                - np.sum(
                    np.arange(2, k.shape[0] + 2)
                    * k
                    * np.power(y[0], np.arange(1, k.shape[0] + 1))
                )
                / m
            )
            return np.array([dx, dv])

        t_span: Tuple[float, float] = (self.t_eval[0], self.t_eval[-1])
        y0 = np.array([self.x0_dist.rvs(), self.v0_dist.rvs()])

        sol = solve_ivp(equation_of_motion, t_span, y0, t_eval=self.t_eval)

        return self.check_solution(sol)


class Inferrer:
    def __init__(
        self,
        t_eval: np.ndarray,
        prior: Any,
        physics_model: BasePhysicsModel,
    ) -> None:
        self.physics_model = physics_model
        self.prior = prior
        self.posterior: Union[None, Any] = None

    def train(self, num_simulations: int = 1000, num_workers: int = 1) -> None:
        def simulator(
            x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
            return self.physics_model.forward_model(np.array(x)).flatten()

        self.posterior = infer(
            simulator,
            self.prior,
            method="SNPE",
            num_simulations=num_simulations,
            num_workers=1,
        )

    def sample_posterior(self, data: torch.Tensor, num_samples: int) -> torch.Tensor:
        if self.posterior is None:
            self.train()

        if self.posterior is not None:
            samples = self.posterior.sample((num_samples,), x=data)

        return samples

    def plot_posterior(
        self,
        data: torch.Tensor,
        true_theta: torch.Tensor,
        names: List[str],
        num_samples: int = 1000,
        param_limits: Any = None,
    ) -> None:
        samples = self.sample_posterior(data, num_samples)

        getdistsamples = getdist.MCSamples(
            samples=samples.cpu().numpy(),
            names=names,
            # labels=self.theta_labels,
            # ranges=self.theta_ranges,
        )

        g = plots.get_single_plotter()
        g.triangle_plot(
            getdistsamples,
            # params = inf.theta_parameters.keys(),
            filled=True,
            markers=true_theta.cpu(),
            param_limits=param_limits,
        )
