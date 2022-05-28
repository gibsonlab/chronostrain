from torch._six import inf
from torch.optim.optimizer import Optimizer

from chronostrain.util.benchmarking import CyclicBuffer


class ReduceLROnPlateauLast(object):
    """
    An adaptation of pytorch's native ReduceLROnPlateau.
    Instead of comparing to the best objective value, compares to the previous observed value instead.
    (To better allow the objective to explore/possibly make the objective temporarily worse in favor of finding a "better" solution.)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience_ratio=0.5, patience_horizon=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.prev = None
        self.bad_epochs = CyclicBuffer(capacity=patience_horizon)
        self.patience_ratio = patience_ratio
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.prev = self.mode_worse
        self.cooldown_counter = 0
        self.bad_epochs.clear()

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if self.is_better(current, self.prev):
            self.bad_epochs.push(0)
        else:
            self.bad_epochs.push(1)

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.bad_epochs.clear()  # ignore any bad epochs in cooldown

        if self.bad_epochs.size == self.bad_epochs.capacity and self.bad_epochs.mean() >= self.patience_ratio:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.bad_epochs.clear()

        self.prev = current

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, prev):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < prev * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < prev - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > prev * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > prev + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)
