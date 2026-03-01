"""Quaternion regularization losses."""

__all__ = ["QuaternionRegularizer"]

import torch


class QuaternionRegularizer:
    """Regularization loss that penalizes non-unit quaternion outputs.

    Args:
        modules: list of nn.Module instances whose outputs are captured via hooks.
        reg_unit: weight for the unit-norm regularization term.
    """

    def __init__(self, modules: list, reg_unit: float = 0.0):
        self.modules = modules
        self.reg_unit = reg_unit
        self._hooks: list = []
        self._captured: torch.Tensor | None = None

    def _hook_fn(self, module, input, output):
        if type(output) is torch.Tensor:
            self._captured = output
        else:
            self._captured = output[0]

    def setup(self, trainer):
        """Register forward hooks on the target modules."""
        for m in self.modules:
            self._hooks.append(m.register_forward_hook(self._hook_fn))

    def teardown(self, trainer):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, pred: torch.Tensor, yb: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """Compute unit-norm regularization loss from captured hook output."""
        if self._captured is None or self.reg_unit == 0.0:
            return torch.tensor(0.0, device=pred.device)

        h = self._captured.float()
        l_a = float(self.reg_unit) * ((1 - h.norm(dim=-1)) ** 2).mean()
        return l_a
