"""dynoNet linear-dynamical-operator model, with its private fused all-pole kernel.

The model lives in :mod:`.core`; :mod:`.allpole_triton` is the fused all-pole IIR
denominator kernel it dispatches to directly.
"""

from .core import *  # noqa: F401,F403
from .core import __all__, _linear_recurrence_sequential  # noqa: F401
