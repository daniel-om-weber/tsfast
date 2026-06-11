"""Training schedule functions."""

import math
from functools import partial

from torch.optim.lr_scheduler import LambdaLR

__all__ = [
    "sched_flat_cos",
    "sched_lin_p",
    "sched_ramp",
    "flat_cos_scheduler",
]


def sched_flat_cos(pos: float, pct_start: float = 0.75) -> float:
    """Flat (1.0) until pct_start, then cosine decay to zero."""
    if pos < pct_start:
        return 1.0
    progress = (pos - pct_start) / (1.0 - pct_start)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _flat_cos_factor(step: int, total_steps: int, pct_start: float) -> float:
    return sched_flat_cos(step / total_steps, pct_start)


def flat_cos_scheduler(opt, total_steps: int, pct_start: float = 0.75) -> LambdaLR:
    """LambdaLR with a flat-then-cosine schedule, built from picklable parts.

    The schedule lambda is a partial of a module-level function (not a closure),
    so a Learner holding this scheduler can be pickled by ``Learner.save``.
    """
    return LambdaLR(opt, partial(_flat_cos_factor, total_steps=total_steps, pct_start=pct_start))


def sched_lin_p(start: float, end: float, pos: float, p: float = 0.75) -> float:
    """Linear schedule that reaches the end value at position p.

    Args:
        start: value at position 0
        end: value at position p and beyond
        pos: current position in [0, 1]
        p: position at which the end value is reached
    """
    return end if pos >= p else start + pos / p * (end - start)


def sched_ramp(start: float, end: float, pos: float, p_left: float = 0.2, p_right: float = 0.6) -> float:
    """Ramp schedule that linearly transitions between two plateau regions.

    Args:
        start: value before p_left
        end: value after p_right
        pos: current position in [0, 1]
        p_left: position where the ramp begins
        p_right: position where the ramp ends
    """
    if pos >= p_right:
        return end
    elif pos <= p_left:
        return start
    else:
        return start + (end - start) * (pos - p_left) / (p_right - p_left)
