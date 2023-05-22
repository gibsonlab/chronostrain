from abc import ABC

import jax.numpy as np
import optax


class LearningRateScheduler(ABC):
    def get_current_lr(self) -> float:
        raise NotImplementedError()

    def step(self, obj_metric: float):
        raise NotImplementedError()

    def get_optax_scheduler(self) -> optax.Schedule:
        # Silly, but helpful in case we decide to switch abstraction from optax to something else
        def _schedule(step_num):
            return self.get_current_lr()
        return _schedule


class ConstantLearningRate(LearningRateScheduler):
    def __init__(self, lr: float):
        self.lr = lr

    def get_current_lr(self) -> float:
        return self.lr

    def step(self, obj_metric: float):
        pass  # do nothing


class ReduceLROnPlateauLast(LearningRateScheduler):
    """
    An adaptation of pytorch's implementation of ReduceLROnPlateau, to be used in optax/jax.
    """
    def __init__(
            self,
            initial_lr: float,
            mode='min',
            factor=0.1,
            patience=10,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-8
    ):
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.lr = initial_lr
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = np.inf
        else:  # mode == 'max':
            self.mode_worse = -np.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _reduce_lr(self):
        new_lr = max(self.lr * self.factor, self.min_lr)
        if self.lr - new_lr > self.eps:
            self.lr = new_lr

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def get_current_lr(self) -> float:
        return self.lr

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            if best > 0:
                rel_epsilon = 1. - self.threshold
            else:
                rel_epsilon = 1 + self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            if best > 0:
                rel_epsilon = self.threshold + 1.
            else:
                rel_epsilon = 1 - self.threshold  # negative values improve the other way.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
