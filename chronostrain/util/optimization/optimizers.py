"""
optimizers.py

Provide wrappers for the appropriate implementation (now JAX, used to be native pytorch).
"""
from typing import *

import optax
import jax.numpy as np
from .scheduler import LearningRateScheduler


class LossOptimizer:
    def __init__(
            self,
            optax_optim: Callable,
            lr_scheduler: LearningRateScheduler,
            hyperparameters: Dict[str, Any],
            minimize_objective: bool = True,
    ):
        self.scheduler = lr_scheduler
        self.optim = optax_optim(learning_rate=lr_scheduler.get_optax_scheduler(), **hyperparameters)
        self.params = None
        self.state = None
        self.grad_sign = 1 if minimize_objective else -1

    def initialize(self, initial_params: Dict[str, np.ndarray]):
        self.params = initial_params
        self.state = self.optim.init(initial_params)

    def current_learning_rate(self) -> float:
        return self.scheduler.get_current_lr()

    def update(self, grad: Dict[Any, np.ndarray]):
        if self.params is None or self.state is None:
            raise RuntimeError("Loss optimizer must be initialized before running.")

        for k, w in grad.items():
            grad[k] = self.grad_sign * grad[k]
        updates, new_opt_state = self.optim.update(grad, self.state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        self.state = new_opt_state
