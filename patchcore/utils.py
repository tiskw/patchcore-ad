"""
Utility functions for the PatchCore
"""

# Import standard libraries.
import time

# Import third-party packages.
import numpy as np


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


def auto_threshold(scores_good, coef_sigma):
    """
    Compute threshold value from the given good scores.

    Args:
        score_good (list) : List of anomaly score for good samples.
        coef_sigma (float): Hyperparameter of the thresholding.
    """
    # Compute mean/std of the anomaly scores.
    score_mean = np.mean(scores_good)
    score_std  = np.std(scores_good)

    # Compute threshold.
    thresh = score_mean + coef_sigma * score_std

    return (thresh, score_mean, score_std)
