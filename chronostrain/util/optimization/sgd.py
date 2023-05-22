from optax import sgd
from .optimizers import LossOptimizer
from .scheduler import LearningRateScheduler


class SGD(LossOptimizer):
    def __init__(self,
                 lr_scheduler: LearningRateScheduler,
                 minimize_objective: bool = True,
                 momentum: float = None,
                 nesterov: bool = True):
        super().__init__(
            sgd,
            minimize_objective=minimize_objective,
            lr_scheduler=lr_scheduler,
            hyperparameters={
                'nesterov': nesterov,
                'momentum': momentum,
            }
        )
