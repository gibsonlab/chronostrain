from optax import adagrad
from .optimizers import LossOptimizer
from .scheduler import LearningRateScheduler


class AdaGrad(LossOptimizer):
    def __init__(self,
                 lr_scheduler: LearningRateScheduler,
                 minimize_objective: bool = True,
                 initial_accumulator_value: float = 0.1,
                 eps: float = 1e-07):
        super().__init__(
            adagrad,
            minimize_objective=minimize_objective,
            lr_scheduler=lr_scheduler,
            hyperparameters={
                'initial_accumulator_value': initial_accumulator_value,
                'eps': eps
            }
        )
