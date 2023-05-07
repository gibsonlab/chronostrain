from jax.example_libraries.optimizers import adam
from .optimizers import LossOptimizer
from .scheduler import LearningRateScheduler


class Adam(LossOptimizer):
    def __init__(self,
                 lr_scheduler: LearningRateScheduler,
                 minimize_objective: bool = True,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 weight_decay: float = 1e-4):
        super().__init__(
            adam,
            lr_scheduler=lr_scheduler,
            minimize_objective=minimize_objective,
            hyperparameters={
                'beta1': beta1,
                'beta2': beta2,
                'eps': eps,
                'weight_decay': weight_decay
            }
        )
