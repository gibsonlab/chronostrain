from .optimizers import LossOptimizer
from .scheduler import LearningRateScheduler
from .scalable_shampoo_google import distributed_shampoo


class DistributedShampoo(LossOptimizer):
    def __init__(self,
                 lr_scheduler: LearningRateScheduler,
                 minimize_objective: bool = True,
                 block_size: int = 128,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 diagonal_epsilon: float = 1e-7,
                 matrix_epsilon: float = 1e-6,
                 weight_decay: float = 0.0,
                 start_preconditioning_step: int = 5,
                 preconditioning_compute_steps: int = 1,
                 nesterov: bool = True,
                 exponent_override: float = 0
                 ):
        super().__init__(
            distributed_shampoo,
            minimize_objective=minimize_objective,
            lr_scheduler=lr_scheduler,
            hyperparameters={
                'block_size': block_size,
                'beta1': beta1,
                'beta2': beta2,
                'diagonal_epsilon': diagonal_epsilon,
                'matrix_epsilon': matrix_epsilon,
                'weight_decay': weight_decay,
                'start_preconditioning_step': start_preconditioning_step,
                'preconditioning_compute_steps': preconditioning_compute_steps,
                'nesterov': nesterov,
                'exponent_override': exponent_override
            }
        )
