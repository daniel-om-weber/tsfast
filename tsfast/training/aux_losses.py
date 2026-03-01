"""Auxiliary loss callables for training."""

__all__ = [
    "AddLoss",
    "PhysicsLoss",
    "TransitionSmoothness",
    "TimeSeriesRegularizerLoss",
    "FranSysRegularizer",
    "ConsistencyLoss",
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


class AddLoss:
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


class PhysicsLoss:
    """Auxiliary loss using a physics-informed loss function.

    Args:
        physics_loss_func: function(u, y_pred, y_ref) returning dict of losses or single tensor
        weight: global scaling factor
        loss_weights: per-component weights like {'physics': 1.0, 'derivative': 0.1}
        n_inputs: number of input features (to slice xb)
        n_skip: initial timesteps to skip
    """

    def __init__(
        self,
        physics_loss_func: Callable,
        weight: float = 1.0,
        loss_weights: dict | None = None,
        n_inputs: int | None = None,
        n_skip: int = 0,
    ):
        self.physics_loss_func = physics_loss_func
        self.weight = weight
        self.loss_weights = loss_weights or {}
        self.n_inputs = n_inputs
        self.n_skip = n_skip

    def __call__(self, pred: Tensor, yb: Tensor, xb: Tensor) -> Tensor:
        u = xb[:, :, : self.n_inputs] if self.n_inputs is not None else xb

        y_pred = pred
        if self.n_skip > 0:
            u = u[:, self.n_skip :]
            y_pred = y_pred[:, self.n_skip :]

        loss_dict = self.physics_loss_func(u, y_pred, None)

        if isinstance(loss_dict, dict):
            total = sum(self.loss_weights.get(k, 1.0) * v for k, v in loss_dict.items())
        else:
            total = loss_dict

        return self.weight * total


class TransitionSmoothness:
    """Penalizes curvature at the init-to-prognosis transition boundary.

    Args:
        init_sz: init window size (transition at this index)
        weight: loss weight
        window: timesteps around boundary to penalize
        dt: time step for derivative computation
    """

    def __init__(self, init_sz: int, weight: float = 1.0, window: int = 3, dt: float = 0.01):
        self.init_sz = init_sz
        self.weight = weight
        self.window = window
        self.dt = dt

    def __call__(self, pred: Tensor, yb: Tensor, xb: Tensor) -> Tensor:
        y = pred
        start = max(0, self.init_sz - self.window)
        end = min(y.shape[1], self.init_sz + self.window)
        if end - start < 3:
            return torch.tensor(0.0, device=y.device)

        y_boundary = y[:, start:end, :]
        batch, wlen, ny = y_boundary.shape
        y_flat = y_boundary.permute(0, 2, 1).reshape(batch * ny, wlen)
        d2 = _diff2_forward(y_flat, self.dt)
        smooth_loss = (d2**2).mean()

        return self.weight * smooth_loss


def _diff2_forward(signal: Tensor, dt: float) -> Tensor:
    """Second-order forward difference: f''(x) ~ (f(x+2h) - 2f(x+h) + f(x)) / h^2."""
    interior = (signal[:, 2:] - 2.0 * signal[:, 1:-1] + signal[:, :-2]) / (dt * dt)
    return torch.cat([interior[:, :1], interior, interior[:, -1:]], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
#  Stateful aux losses — classes with setup(trainer) / teardown(trainer)
# ──────────────────────────────────────────────────────────────────────────────


class TimeSeriesRegularizerLoss:
    """Activation regularization (AR) and temporal activation regularization (TAR).

    Args:
        modules: modules to hook for capturing activations
        alpha: coefficient for AR penalty (L2 on activations)
        beta: coefficient for TAR penalty (L2 on consecutive activation differences)
        dim: time axis index; auto-detected from the hooked layer output if None
    """

    def __init__(
        self,
        modules: list[nn.Module],
        alpha: float = 0.0,
        beta: float = 0.0,
        dim: int | None = None,
    ):
        self.modules = modules
        self.alpha = alpha
        self.beta = beta
        self.dim = dim
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._out: Tensor | None = None

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

    def __call__(self, pred: Tensor, yb: Tensor, xb: Tensor) -> Tensor:
        if self._out is None:
            return torch.tensor(0.0, device=pred.device)

        h = self._out.float()
        loss = torch.tensor(0.0, device=pred.device)

        if self.alpha != 0.0:
            loss = loss + float(self.alpha) * h.pow(2).mean()

        if self.beta != 0.0 and h.shape[self.dim] > 1:
            h_diff = (h[:, 1:] - h[:, :-1]) if self.dim == 1 else (h[:, :, 1:] - h[:, :, :-1])
            loss = loss + float(self.beta) * h_diff.pow(2).mean()

        self._out = None
        return loss


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

    def __call__(self, pred: Tensor, yb: Tensor, xb: Tensor) -> Tensor:
        if self._out_diag is None or self._out_prog is None:
            self._clear()
            return torch.tensor(0.0, device=pred.device)

        diag = self._out_diag
        prog = self._out_prog
        self._clear()
        model = self.inner_model
        win_reg = (
            self.osp_n_skip if self.osp_n_skip is not None else getattr(model, "_effective_init_sz", model.init_sz)
        )

        diag_trunc = diag
        if diag.shape[2] > prog.shape[2]:
            diag_trunc = diag_trunc[:, :, -prog.shape[2] :]

        loss = torch.tensor(0.0, device=pred.device)

        # sync diag prog hidden states loss
        if self.p_state_sync > 0:
            if self.sync_type == "mse":
                hidden_loss = ((prog - diag_trunc) / (diag_trunc.norm())).pow(2).mean()
            elif self.sync_type == "mae":
                hidden_loss = ((prog - diag_trunc) / (diag_trunc.norm())).abs().mean()
            elif self.sync_type == "mspe":
                hidden_loss = (
                    ((diag_trunc - prog) / torch.linalg.norm(diag_trunc, dim=(0, 1), keepdim=True)).pow(2).mean()
                )
            elif self.sync_type == "mape":
                hidden_loss = (
                    ((diag_trunc - prog) / torch.linalg.norm(diag_trunc, dim=(0, 1), keepdim=True)).abs().mean()
                )
            elif self.sync_type == "cos":
                hidden_loss = cos_sim_loss(diag_trunc, prog)
            elif self.sync_type == "cos_pow":
                hidden_loss = cos_sim_loss_pow(diag_trunc, prog)
            else:
                raise ValueError(f"Unknown sync_type: {self.sync_type}")

            loss = loss + self.p_state_sync * hidden_loss

        # diagnosis loss
        if self.p_diag_loss > 0:
            y_diag = model.final(diag_trunc[-1])
            if self._output_norm is not None:
                y_diag = self._output_norm.denormalize(y_diag)
            hidden_loss = self.targ_loss_func(y_diag, yb[:, -y_diag.shape[1] :])
            loss = loss + self.p_diag_loss * hidden_loss

        # osp loss - one step prediction on every element of the sequence
        if self.p_osp_loss > 0 or self.p_osp_sync > 0:
            inp = xb[:, win_reg:]
            bs, n, _ = inp.shape
            inp = torch.flatten(inp[:, :, : model.n_u], start_dim=0, end_dim=1)[:, None, :]
            h_init = torch.flatten(diag[:, :, win_reg - 1 : -1], start_dim=1, end_dim=2)[:, None]

            out, _ = model.rnn_prognosis(inp, h_init)
            h_out = out[:, :, 0]
            out = out[-1].unflatten(0, (bs, n))[:, :, 0]

            # osp hidden sync loss
            h_out_targ = torch.flatten(diag[:, :, win_reg:], start_dim=1, end_dim=2)
            hidden_loss = ((h_out_targ - h_out) / (h_out.norm() + h_out_targ.norm())).pow(2).mean()
            loss = loss + self.p_osp_sync * hidden_loss

            # osp target loss
            y_osp = model.final(out)
            if self._output_norm is not None:
                y_osp = self._output_norm.denormalize(y_osp)
            hidden_loss = self.targ_loss_func(y_osp, yb[:, -y_osp.shape[1] :])
            loss = loss + self.p_osp_loss * hidden_loss

        # tar hidden loss
        if self.p_tar_loss > 0:
            h = torch.cat([diag[:, :, : getattr(model, "_effective_init_sz", model.init_sz)], prog], 2)
            h_diff = h[:, :, 1:] - h[:, :, :-1]
            hidden_loss = h_diff.pow(2).mean()
            loss = loss + self.p_tar_loss * hidden_loss

        return loss


class ConsistencyLoss:
    """Trains SequenceEncoder and StateEncoder compatibility on real data.

    Hooks ``rnn_diagnosis`` and computes MSE between sequence encoder hidden
    state and state encoder hidden state at a given timestep.

    Args:
        weight: scaling factor for consistency loss
        match_at_timestep: timestep to match (defaults to model.init_sz - 1)
        model: explicit inner model reference (auto-detected via unwrap_model if None)
    """

    def __init__(
        self,
        weight: float = 1.0,
        match_at_timestep: int | None = None,
        model: nn.Module | None = None,
    ):
        self.weight = weight
        self.match_at_timestep = match_at_timestep
        self.inner_model = model
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._diag_out: Tensor | None = None

    def setup(self, trainer):
        from ..models.layers import unwrap_model

        if self.inner_model is None:
            self.inner_model = unwrap_model(trainer.model)
        if hasattr(self.inner_model, "rnn_diagnosis") and hasattr(self.inner_model, "encode_single_state"):
            self._hooks.append(self.inner_model.rnn_diagnosis.register_forward_hook(self._hook_fn))

    def teardown(self, trainer):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _hook_fn(self, m, i, o):
        self._diag_out = o[0]

    def __call__(self, pred: Tensor, yb: Tensor, xb: Tensor) -> Tensor:
        if self._diag_out is None:
            return torch.tensor(0.0, device=pred.device)

        model = self.inner_model
        timestep = self.match_at_timestep
        if timestep is None and hasattr(model, "init_sz"):
            timestep = getattr(model, "_effective_init_sz", model.init_sz) - 1
        elif timestep is None:
            timestep = -1

        h_sequence = model.rnn_diagnosis.output_to_hidden(self._diag_out, timestep)
        physical_state = yb[:, timestep, :]
        h_state = model.encode_single_state(physical_state)

        loss = torch.tensor(0.0, device=pred.device)
        for h_seq, h_st in zip(h_sequence, h_state):
            loss = loss + F.mse_loss(h_seq, h_st)

        self._diag_out = None
        return self.weight * loss
