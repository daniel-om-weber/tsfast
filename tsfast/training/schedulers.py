"""Training schedule functions."""

__all__ = [
    "sched_lin_p",
    "sched_ramp",
]


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
