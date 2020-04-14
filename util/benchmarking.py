"""
  benchmarking.py
  Contains helper functions for benchmarking.
"""

import time


def current_time_seconds():
    return int(time.time())


def seconds_elapsed(start):
    return current_time_seconds() - start


def minutes_elapsed(start):
    return int(seconds_elapsed(start) / 60)


class CyclicIntegerBuffer:
    """
    An (integer-valued) buffer of finite capcity. Cyclically overwrites next available slot in memory.
    """
    def __init__(self, size, default_value=0):
        self.size = size
        self.buf = [default_value for _ in range(size)]
        self.total = 0
        self.next_idx = 0

    def push(self, val):
        self.total += val - self.buf[self.next_idx]
        self.buf[self.next_idx] = val
        self.next_idx += 1

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
        self.iter_buf = CyclicIntegerBuffer(size=horizon)
        self.last_click = None

    def stopwatch_click(self):
        """
        Registers a click on the stopwatch (starts recording a new lap).
        :return: The time elapsed since last click, in seconds.
        """

        if self.last_click is None:
            self.last_click = current_time_seconds()
            return None
        else:
            now = current_time_seconds()
            diff = seconds_elapsed(now)
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

