"""
  benchmarking.py
  Contains helper functions for benchmarking.
"""

import time


def current_time_millis():
    return int(round(time.time() * 1000))


def millis_elapsed(start_millis):
    return current_time_millis() - start_millis


class CyclicBuffer:
    """
    A buffer of finite capcity. Cyclically overwrites next available slot in memory.
    """
    def __init__(self, capacity):
        if capacity == 0:
            raise ValueError("Buffer size must be nonzero.")
        self.capacity = capacity
        self.size = 0
        self.buf = [0] * capacity
        self.total = 0
        self.next_idx = 0

    def push(self, val):
        self.total += val - self.buf[self.next_idx]
        self.buf[self.next_idx] = val
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def values(self):
        return self.buf

    def sum(self):
        return self.total

    def mean(self):
        return self.total / self.size


class RuntimeEstimator:
    """
    A class that keeps track of runtime.
    """
    def __init__(self, total_iters, horizon=5):
        """
        :param total_iters: The total expected number of iterations.
        :param horizon: The number of iterations to average over (to estimate speed of program).
        """
        self.total_iters = total_iters
        self.iters_left = total_iters
        self.iter_buf = CyclicBuffer(capacity=horizon)
        self.last_click = None

    def stopwatch_click(self):
        """
        Registers a click on the stopwatch (starts recording a new lap).
        :return: The time elapsed since last click, in milliseconds.
        """

        if self.last_click is None:
            self.last_click = current_time_millis()
            return None
        else:
            now = current_time_millis()
            diff = millis_elapsed(self.last_click)
            self.last_click = now
            return diff

    def increment(self, iter_runtime):
        """
        Record the amount of time taken for one iteration.
        :param iter_runtime:
        :return:
        """
        self.iter_buf.push(iter_runtime)
        self.iters_left -= 1

    def time_left(self):
        return self.iters_left * self.iter_buf.mean()
