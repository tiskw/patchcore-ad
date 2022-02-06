"""
Utility functions for the PatchCore
"""

import time


class Timer:
    """
    A class for measuring elapsed time by "with" sentence.

    Example:
    >>> # Create Timer instance.
    >>> timer = Timer()
    >>> 
    >>> # Repeat some procedure 100 times.
    >>> for _ in range(100):
    >>>     with timer:
    >>>         some_procedure()
    >>> 
    >>> # Print mean elapsed time.
    >>> print(timer.mean())
    """
    def __init__(self):
        self.times = list()

    def __enter__(self):
        self.time_start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.time_end = time.time()
        self.times.append(self.time_end - self.time_start)

    def mean(self):
        return sum(self.times) / len(self.times)
