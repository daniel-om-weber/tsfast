"""Auxiliary loss callables for training."""

__all__ = [
    "AuxiliaryLoss",
    "ActivationRegularizer",
    "TemporalActivationRegularizer",
    "FranSysRegularizer",
]

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .losses import cos_sim_loss, cos_sim_loss_pow


# ──────────────────────────────────────────────────────────────────────────────
#  Simple aux loss callables — __call__(pred, yb, xb) -> loss_term
# ──────────────────────────────────────────────────────────────────────────────


class AuxiliaryLoss:
    """Auxiliary loss that applies a loss function to predictions and targets.

    Args:
        loss_func: loss function applied to (pred, targ)
        alpha: scaling factor for the auxiliary loss
    """

    def __init__(self, loss_func: Callable, alpha: float = 1.0):
        self.loss_func = loss_func
        self.alpha = alpha

    def __call__(self, pred: Tensor, yb: Tensor, xb: Tensor) -> Tensor:
        return self.alpha * self.loss_func(pred, yb)


# ──────────────────────────────────────────────────────────────────────────────
#  Stateful aux losses — classes with setup(trainer) / teardown(trainer)
# ──────────────────────────────────────────────────────────────────────────────


class _ActivationHookMixin:
    """Shared hook logic for activation-based regularizers."""

    def setup(self, trainer):
        for m in self.modules:
            self._hooks.append(m.register_forward_hook(self._hook_fn))

    def teardown(self, trainer):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _hook_fn(self, m, i, o):
        if isinstance(o, torch.Tensor):
            self._out = o
        else:
            self._out = o[0]

        if self.dim is None:
            self.dim = int(np.argmax([0, self._out.shape[1], self._out.shape[2]]))


class ActivationRegularizer(_ActivationHookMixin):
    """L2 penalty on hooked module activations (activation regularization).

    Args:
        modules: modules to hook for capturing activations
        alpha: coefficient for the L2 penalty
        dim: time axis index; auto-detected from the hooked layer output if None
    """

    def __init__(self, modules: list[nn.Module], alpha: float = 1.0, dim: int | None = None):
        self.modules = modules
        self.alpha = alpha
        self.dim = dim
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._out: Tensor | None = None

    def __call__(self, pred: Tensor, yb: Tensor, xb: Tensor) -> Tensor:
        if self._out is None:
            return torch.tensor(0.0, device=pred.device)
        h = self._out.float()
        self._out = None
        return float(self.alpha) * h.pow(2).mean()


class TemporalActivationRegularizer(_ActivationHookMixin):
    """L2 penalty on consecutive-timestep activation differences (temporal activation regularization).

    Args:
        modules: modules to hook for capturing activations
        beta: coefficient for the L2 penalty on temporal differences
        dim: time axis index; auto-detected from the hooked layer output if None
    """

    def __init__(self, modules: list[nn.Module], beta: float = 1.0, dim: int | None = None):
        self.modules = modules
        self.beta = beta
        self.dim = dim
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._out: Tensor | None = None

    def __call__(self, pred: Tensor, yb: Tensor, xb: Tensor) -> Tensor:
        if self._out is None:
            return torch.tensor(0.0, device=pred.device)
        h = self._out.float()
        self._out = None
        if h.shape[self.dim] <= 1:
            return torch.tensor(0.0, device=pred.device)
        h_diff = (h[:, 1:] - h[:, :-1]) if self.dim == 1 else (h[:, :, 1:] - h[:, :, :-1])
        return float(self.beta) * h_diff.pow(2).mean()


class FranSysRegularizer:
    """Regularizes FranSys output by syncing diagnosis and prognosis hidden states.

    Args:
        modules: modules to hook (diagnosis + prognosis RNNs)
        p_state_sync: scaling factor for hidden state sync loss
        p_diag_loss: scaling factor for diagnosis loss through the final layer
        p_osp_sync: scaling factor for one-step prediction hidden state sync
        p_osp_loss: scaling factor for one-step prediction loss
        p_tar_loss: scaling factor for temporal activation regularization
        sync_type: distance metric for state synchronization
        targ_loss_func: loss function for target-based regularization
        osp_n_skip: elements to skip before one-step prediction (defaults to model.init_sz)
        model: explicit FranSys model reference (auto-detected via unwrap_model if None)
    """

    _SYNC_FNS: dict[str, Callable] = {
        "mse": lambda d, p: ((p - d) / d.norm()).pow(2).mean(),
        "mae": lambda d, p: ((p - d) / d.norm()).abs().mean(),
        "mspe": lambda d, p: ((d - p) / torch.linalg.norm(d, dim=(0, 1), keepdim=True)).pow(2).mean(),
        "mape": lambda d, p: ((d - p) / torch.linalg.norm(d, dim=(0, 1), keepdim=True)).abs().mean(),
        "cos": lambda d, p: cos_sim_loss(d, p),
        "cos_pow": lambda d, p: cos_sim_loss_pow(d, p),
    }

    def __init__(
        self,
        modules: list[nn.Module],
        p_state_sync: float = 1e7,
        p_diag_loss: float = 0.0,
        p_osp_sync: float = 0,
        p_osp_loss: float = 0,
        p_tar_loss: float = 0,
        sync_type: str = "mse",
        targ_loss_func: Callable = F.l1_loss,
        osp_n_skip: int | None = None,
        model: nn.Module | None = None,
    ):
        if sync_type not in self._SYNC_FNS:
            raise ValueError(f"Unknown sync_type: {sync_type!r}")
        self.modules = modules
        self.p_state_sync = p_state_sync
        self.p_diag_loss = p_diag_loss
        self.p_osp_sync = p_osp_sync
        self.p_osp_loss = p_osp_loss
        self.p_tar_loss = p_tar_loss
        self.sync_type = sync_type
        self.targ_loss_func = targ_loss_func
        self.osp_n_skip = osp_n_skip
        self.inner_model = model
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._out_diag: Tensor | None = None
        self._out_prog: Tensor | None = None
        self._output_norm = None

    def setup(self, trainer):
        from ..models.layers import NormalizedModel, _unwrap_ddp, unwrap_model

        wrapper = _unwrap_ddp(trainer.model)
        self._output_norm = wrapper.output_norm if isinstance(wrapper, NormalizedModel) else None
        if self.inner_model is None:
            self.inner_model = unwrap_model(trainer.model)
        for m in self.modules:
            self._hooks.append(m.register_forward_hook(self._hook_fn))

    def teardown(self, trainer):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _hook_fn(self, m, i, o):
        if "Diag" in type(m).__name__:
            self._out_diag = o[0]
        else:
            self._out_prog = o[0]

    def _clear(self):
        self._out_diag = None
        self._out_prog = None

    def _state_sync_loss(self, diag_trunc: Tensor, prog: Tensor) -> Tensor:
        if self.p_state_sync <= 0:
            return torch.tensor(0.0, device=prog.device)
        return self.p_state_sync * self._SYNC_FNS[self.sync_type](diag_trunc, prog)

    def _diag_loss(self, diag_trunc: Tensor, yb: Tensor, model: nn.Module) -> Tensor:
        if self.p_diag_loss <= 0:
            return torch.tensor(0.0, device=yb.device)
        y_diag = model.final(diag_trunc[-1])
        if self._output_norm is not None:
            y_diag = self._output_norm.denormalize(y_diag)
        return self.p_diag_loss * self.targ_loss_func(y_diag, yb[:, -y_diag.shape[1] :])

    def _osp_loss(self, diag: Tensor, xb: Tensor, yb: Tensor, model: nn.Module, win_reg: int) -> Tensor:
        if self.p_osp_loss <= 0 and self.p_osp_sync <= 0:
            return torch.tensor(0.0, device=xb.device)

        inp = xb[:, win_reg:]
        bs, n, _ = inp.shape
        inp = torch.flatten(inp[:, :, : model.n_u], start_dim=0, end_dim=1)[:, None, :]
        h_init = torch.flatten(diag[:, :, win_reg - 1 : -1], start_dim=1, end_dim=2)[:, None]

        out, _ = model.rnn_prognosis(inp, h_init)
        h_out = out[:, :, 0]
        out = out[-1].unflatten(0, (bs, n))[:, :, 0]

        loss = torch.tensor(0.0, device=xb.device)

        # hidden sync
        h_out_targ = torch.flatten(diag[:, :, win_reg:], start_dim=1, end_dim=2)
        loss = loss + self.p_osp_sync * ((h_out_targ - h_out) / (h_out.norm() + h_out_targ.norm())).pow(2).mean()

        # target loss
        y_osp = model.final(out)
        if self._output_norm is not None:
            y_osp = self._output_norm.denormalize(y_osp)
        loss = loss + self.p_osp_loss * self.targ_loss_func(y_osp, yb[:, -y_osp.shape[1] :])

        return loss

    def _tar_loss(self, diag: Tensor, prog: Tensor, model: nn.Module) -> Tensor:
        if self.p_tar_loss <= 0:
            return torch.tensor(0.0, device=prog.device)
        h = torch.cat([diag[:, :, : getattr(model, "_effective_init_sz", model.init_sz)], prog], 2)
        h_diff = h[:, :, 1:] - h[:, :, :-1]
        return self.p_tar_loss * h_diff.pow(2).mean()

    def __call__(self, pred: Tensor, yb: Tensor, xb: Tensor) -> Tensor:
        if self._out_diag is None or self._out_prog is None:
            self._clear()
            return torch.tensor(0.0, device=pred.device)

        diag, prog = self._out_diag, self._out_prog
        self._clear()
        model = self.inner_model
        win_reg = (
            self.osp_n_skip if self.osp_n_skip is not None else getattr(model, "_effective_init_sz", model.init_sz)
        )
        diag_trunc = diag[:, :, -prog.shape[2] :] if diag.shape[2] > prog.shape[2] else diag

        return (
            self._state_sync_loss(diag_trunc, prog)
            + self._diag_loss(diag_trunc, yb, model)
            + self._osp_loss(diag, xb, yb, model, win_reg)
            + self._tar_loss(diag, prog, model)
        )
