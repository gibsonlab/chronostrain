from optax import adam
from .optimizers import LossOptimizer
from .scheduler import LearningRateScheduler


class Adam(LossOptimizer):
    def __init__(self,
                 lr_scheduler: LearningRateScheduler,
                 minimize_objective: bool = True,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8):
        super().__init__(
            adam,
            minimize_objective=minimize_objective,
            lr_scheduler=lr_scheduler,
            hyperparameters={
                'b1': beta1,
                'b2': beta2,
                'eps': eps
            }
        )
