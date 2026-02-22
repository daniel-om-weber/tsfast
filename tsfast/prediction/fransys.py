"""FranSys prediction framework: diagnosis/prognosis models and training utilities."""

__all__ = [
    "Diag_RNN",
    "Diag_RNN_raw",
    "DiagLSTM",
    "Diag_TCN",
    "ARProg_Init",
    "FranSys",
    "FranSysCallback",
    "FranSysCallback_variable_init",
    "FranSysLearner",
]

from functools import partial

import numpy as np
import torch
from torch import nn

from fastcore.imports import is_iter
from fastcore.meta import delegates

from fastai.callback.core import Callback
from fastai.callback.hook import HookCallback
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.metrics import mae
from fastai.optimizer import Adam

from ..learner.callbacks import CB_TruncateSequence
from ..learner.losses import SkipNLoss, cos_sim_loss, cos_sim_loss_pow, fun_rmse
from ..models.cnn import TCN
from ..datasets.core import ensure_norm_stats
from ..models.layers import AR_Model, NormalizedModel, SeqLinear, StandardScaler1D
from ..models.rnn import RNN, SimpleRNN
from .core import PredictionCallback


class Diag_RNN(nn.Module):
    """RNN-based diagnosis model with configurable output layers.

    Args:
        input_size: number of input features
        output_size: number of output features per output layer
        output_layer: number of stacked output layers
        hidden_size: number of hidden units in the RNN
        rnn_layer: number of RNN layers
        linear_layer: number of linear layers in the output head
        stateful: whether to maintain hidden state across batches
    """

    @delegates(RNN, keep=True)
    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_layer: int = 1,
        hidden_size: int = 100,
        rnn_layer: int = 1,
        linear_layer: int = 1,
        stateful: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.output_size = output_size

        self.rnn = RNN(input_size, hidden_size, rnn_layer, stateful=stateful, ret_full_hidden=False, **kwargs)
        self.final = SeqLinear(hidden_size, int(output_size * output_layer), hidden_layer=linear_layer - 1)

    def forward(self, x: torch.Tensor, init_state: list | None = None) -> tuple[torch.Tensor, list]:
        out, hidden = self.rnn(x, init_state)
        out = self.final(out)
        out = torch.stack(torch.split(out, split_size_or_sections=self.output_size, dim=-1), 0)
        return out, hidden

    def output_to_hidden(self, out: torch.Tensor, idx: int) -> list[torch.Tensor]:
        """Extract hidden states from output at a given time index."""
        hidden = list(out[:, None, :, idx])
        #         hidden = torch.split(out[:,:,idx],split_size_or_sections=1,dim = 0)
        hidden = [h.contiguous() for h in hidden]
        return hidden

    def _get_hidden(self, bs):
        return self.rnn._get_hidden(bs)


class Diag_RNN_raw(nn.Module):
    """Raw RNN diagnosis model that returns full hidden states directly.

    Args:
        input_size: number of input features
        output_size: number of output features
        output_layer: number of stacked output layers
        hidden_size: number of hidden units in the RNN
        rnn_layer: number of RNN layers
        linear_layer: number of linear layers (unused, kept for API compatibility)
        stateful: whether to maintain hidden state across batches
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_layer: int = 1,
        hidden_size: int = 100,
        rnn_layer: int = 1,
        linear_layer: int = 1,
        stateful: bool = False,
    ):
        super().__init__()

        self.rnn = RNN(input_size, output_size, output_layer, stateful=stateful, ret_full_hidden=True)

    def forward(self, x: torch.Tensor, init_state: list | None = None):
        return self.rnn(x, init_state)

    def output_to_hidden(self, out: torch.Tensor, idx: int) -> list[torch.Tensor]:
        """Extract hidden states from output at a given time index."""
        hidden = list(out[:, None, :, idx])
        #         hidden = torch.split(out[:,:,idx],split_size_or_sections=1,dim = 0)
        hidden = [h.contiguous() for h in hidden]
        return hidden


class DiagLSTM(nn.Module):
    """LSTM-based diagnosis model with configurable output layers.

    Args:
        input_size: number of input features
        output_size: number of output features per output layer
        output_layer: number of stacked output layers
        hidden_size: number of hidden units in the LSTM
        rnn_layer: number of LSTM layers
        linear_layer: number of linear layers in the output head
    """

    @delegates(nn.LSTM, keep=True)
    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_layer: int = 1,
        hidden_size: int = 100,
        rnn_layer: int = 1,
        linear_layer: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_layer = output_layer
        self.hidden_size = hidden_size
        self.rnn_layer = rnn_layer
        self.linear_layer = linear_layer

        self.rnn = nn.LSTM(input_size, hidden_size, rnn_layer, batch_first=True, **kwargs)
        self.final = SeqLinear(hidden_size, int(output_size * output_layer * 2), hidden_layer=linear_layer - 1)

    def forward(self, x: torch.Tensor, init_state: tuple | None = None) -> torch.Tensor:
        out, _ = self.rnn(x, init_state)
        out = self.final(out)
        # split tensor in n hidden layers of the prognosis rnn
        out = torch.stack(torch.split(out, split_size_or_sections=self.output_size, dim=-1), 0)
        return out

    def output_to_hidden(self, out: torch.Tensor, idx: int) -> list[tuple[torch.Tensor, ...]]:
        """Extract LSTM hidden and cell states from output at a given time index."""
        hidden = out[:, :, idx]
        # split in target rnn layers
        lst_hidden = hidden.split(hidden.shape[0] // self.output_layer)
        # split in tuples for the lstm
        lst_hidden = [tuple([h_split.contiguous() for h_split in h.split(h.shape[0] // 2, dim=0)]) for h in lst_hidden]
        return lst_hidden

    def _get_hidden(self, bs):
        return self.rnn._get_hidden(bs)


class Diag_TCN(nn.Module):
    """TCN-based diagnosis model with optional MLP output head.

    Args:
        input_size: number of input features
        output_size: number of output features per output layer
        output_layer: number of stacked output layers
        hl_width: width of TCN hidden layers
        mlp_layers: number of additional MLP layers after the TCN
    """

    @delegates(TCN, keep=True)
    def __init__(
        self, input_size: int, output_size: int, output_layer: int, hl_width: int, mlp_layers: int = 0, **kwargs
    ):
        super().__init__()
        self.output_size = output_size

        if mlp_layers > 0:
            self._model = TCN(input_size, hl_width, hl_width=hl_width, **kwargs)
            self.final = SeqLinear(
                hl_width, int(output_size * output_layer), hidden_size=hl_width, hidden_layer=mlp_layers
            )
        else:
            self._model = TCN(input_size, int(output_size * output_layer), hl_width=hl_width, **kwargs)
            self.final = nn.Identity()

    def forward(self, x: torch.Tensor, init_state: list | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._model(x)
        out = self.final(out)
        out = torch.stack(torch.split(out, split_size_or_sections=self.output_size, dim=-1), 0)
        return out, out

    def output_to_hidden(self, out: torch.Tensor, idx: int) -> list[torch.Tensor]:
        """Extract hidden states from output at a given time index."""
        hidden = list(out[:, None, :, idx])
        #         hidden = torch.split(out[:,:,idx],split_size_or_sections=1,dim = 0)
        hidden = [h.contiguous() for h in hidden]
        return hidden


class ARProg_Init(nn.Module):
    """Autoregressive prognosis model with diagnosis-based initialization.

    Args:
        n_u: number of input channels
        n_y: number of output channels
        init_sz: number of initial time steps used for diagnosis
        n_x: number of external state channels
        hidden_size: number of hidden units in the RNN
        rnn_layer: number of RNN layers
        diag_model: custom diagnosis model, defaults to Diag_RNN
        linear_layer: number of linear layers in the diagnosis output head
        final_layer: number of additional final layers (unused, reserved)
    """

    @delegates(RNN, keep=True)
    def __init__(
        self,
        n_u: int,
        n_y: int,
        init_sz: int,
        n_x: int = 0,
        hidden_size: int = 100,
        rnn_layer: int = 1,
        diag_model: nn.Module | None = None,
        linear_layer: int = 1,
        final_layer: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.init_sz = init_sz
        self.n_x = n_x
        self.hidden_size = hidden_size
        self.rnn_layer = rnn_layer
        self.diag_model = diag_model
        self.linear_layer = linear_layer
        self.final_layer = final_layer

        rnn_kwargs = dict(hidden_size=hidden_size, num_layers=rnn_layer)
        rnn_kwargs = dict(rnn_kwargs, **kwargs)

        if diag_model is None:
            self.rnn_diagnosis = Diag_RNN(
                n_u + n_x + n_y,
                output_size=hidden_size,
                hidden_size=hidden_size,
                output_layer=rnn_layer,
                rnn_layer=rnn_layer,
                linear_layer=linear_layer,
            )
        else:
            self.rnn_diagnosis = diag_model

        self.rnn_prognosis = AR_Model(
            SimpleRNN(input_size=n_u + n_x + n_y, output_size=n_x + n_y, return_state=True, **rnn_kwargs),
            model_has_state=True,
            return_state=False,
            ar=True,
            out_sz=n_x + n_y,
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        y_x = inp[..., self.n_u :]  # measured output and external state
        u = inp[..., : self.n_u]  # measured input

        if self.training:
            out_diag, _ = self.rnn_diagnosis(inp)
        else:
            out_diag, _ = self.rnn_diagnosis(inp[:, : self.init_sz])
        h_init = self.rnn_diagnosis.output_to_hidden(out_diag, self.init_sz - 1)
        self.rnn_prognosis.y_init = y_x[:, self.init_sz : self.init_sz + 1]
        out_prog = self.rnn_prognosis(u[:, self.init_sz :], h_init=h_init, ar=True)

        result = torch.cat([torch.zeros(inp.shape[0], self.init_sz, y_x.shape[2], device=inp.device), out_prog], 1)

        return result[..., -self.n_y :]


class FranSys(nn.Module):
    """Framework for Analysis of Systems: combined diagnosis/prognosis RNN model.

    Args:
        n_u: number of input channels
        n_y: number of output channels
        init_sz: number of initial time steps used for diagnosis
        n_x: number of external state channels
        hidden_size: number of hidden units in the RNN
        rnn_layer: number of RNN layers
        diag_model: custom diagnosis model, defaults to Diag_RNN
        linear_layer: number of linear layers in the diagnosis output head
        init_diag_only: if True, limit diagnosis to init_sz time steps during training
        final_layer: number of additional layers in the shared output head
    """

    @delegates(RNN, keep=True)
    def __init__(
        self,
        n_u: int,
        n_y: int,
        init_sz: int,
        n_x: int = 0,
        hidden_size: int = 100,
        rnn_layer: int = 1,
        diag_model: nn.Module | None = None,
        linear_layer: int = 1,
        init_diag_only: bool = False,
        final_layer: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x
        self.init_sz = init_sz
        self.init_diag_only = init_diag_only

        rnn_kwargs = dict(hidden_size=hidden_size, num_layers=rnn_layer, ret_full_hidden=True)
        rnn_kwargs = dict(rnn_kwargs, **kwargs)

        if diag_model is None:
            self.rnn_diagnosis = Diag_RNN(
                n_u + n_x + n_y,
                hidden_size,
                hidden_size=hidden_size,
                output_layer=rnn_layer,
                rnn_layer=rnn_layer,
                linear_layer=linear_layer,
                **kwargs,
            )
        else:
            self.rnn_diagnosis = diag_model
        self.rnn_prognosis = RNN(n_u, **rnn_kwargs)

        #        self.final = SeqLinear(int(hidden_size*rnn_layer),n_y,hidden_layer=0)
        self.final = SeqLinear(hidden_size, n_y, hidden_layer=final_layer)

    def forward(self, x: torch.Tensor, init_state: list | None = None) -> torch.Tensor:
        x_diag = x[..., : self.n_u + self.n_x + self.n_y]
        x_prog = x[..., : self.n_u]

        if self.init_diag_only:
            x_diag = x_diag[:, : self.init_sz]  # limit diagnosis length to init size

        if self.training:
            # in training, estimate the full sequence with the diagnosis module
            if init_state is None:
                # execution with no initial state
                out_diag, _ = self.rnn_diagnosis(x_diag)
                h_init = self.rnn_diagnosis.output_to_hidden(out_diag, self.init_sz - 1)

                # ToDo: only execute this if callback is used
                new_hidden = self.rnn_diagnosis.output_to_hidden(out_diag, -1)

                out_prog, _ = self.rnn_prognosis(x_prog[:, self.init_sz :].contiguous(), h_init)
                out_prog = torch.cat([out_diag[:, :, : self.init_sz], out_prog], 2)
            else:
                # import pdb; pdb.set_trace()
                out_prog, _ = self.rnn_prognosis(x_prog, init_state)
                out_diag, _ = self.rnn_diagnosis(x_diag)
                new_hidden = self.rnn_diagnosis.output_to_hidden(out_diag, -1)
        else:
            #             import pdb; pdb.set_trace()
            # in inference, use the diagnosis module only for initial state estimation
            if init_state is None:
                out_init, _ = self.rnn_diagnosis(x_diag[:, : self.init_sz])
                h_init = self.rnn_diagnosis.output_to_hidden(out_init, -1)
                out_prog, new_hidden = self.rnn_prognosis(x_prog[:, self.init_sz :], h_init)
                out_prog = torch.cat([out_init, out_prog], 2)
            else:
                out_prog, new_hidden = self.rnn_prognosis(x_prog, init_state)

        # Shared Linear Layer
        result = self.final(out_prog[-1])
        return result


class FranSysCallback(HookCallback):
    """Regularizes FranSys output by syncing diagnosis and prognosis hidden states.

    Args:
        modules: modules to hook into for capturing hidden states
        p_state_sync: scaling factor for hidden state deviation between diagnosis
            and prognosis modules
        p_diag_loss: scaling factor for loss of diagnosis hidden state through
            the final layer
        p_osp_sync: scaling factor for hidden state deviation between one-step
            prediction and diagnosis hidden states
        p_osp_loss: scaling factor for one-step prediction loss of prognosis module
        p_tar_loss: scaling factor for time activation regularization of combined
            hidden states with target sequence length
        sync_type: distance metric for state synchronization loss
        targ_loss_func: loss function used for target-based regularization terms
        osp_n_skip: number of elements to skip before one-step prediction is
            applied, defaults to model.init_sz
        model: explicit FranSys model reference, auto-detected if None
        detach: whether to detach hooked outputs from the computation graph
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
        targ_loss_func=mae,
        osp_n_skip: int | None = None,
        model: nn.Module | None = None,
        detach: bool = False,
        **kwargs,
    ):
        super().__init__(modules=modules, detach=detach, cpu=False, **kwargs)
        self.p_state_sync = p_state_sync
        self.p_diag_loss = p_diag_loss
        self.p_osp_sync = p_osp_sync
        self.p_osp_loss = p_osp_loss
        self.p_tar_loss = p_tar_loss
        self.sync_type = sync_type
        self.targ_loss_func = targ_loss_func
        self.osp_n_skip = osp_n_skip
        self.inner_model = model
        self.clear()

    def before_fit(self):
        from ..models.layers import NormalizedModel, _unwrap_ddp

        wrapper = _unwrap_ddp(self.learn.model)
        self._output_norm = wrapper.output_norm if isinstance(wrapper, NormalizedModel) else None
        if self.inner_model is None:
            from ..models.layers import unwrap_model

            self.inner_model = unwrap_model(self.learn.model)
        super().before_fit()

    def clear(self):
        """Reset captured diagnosis and prognosis outputs."""
        self._out_diag = None
        self._out_prog = None

    def hook(self, m, i, o):
        """Capture output of diagnosis and prognosis modules for regularization in after_loss."""
        if "Diag" in type(m).__name__:
            self._out_diag = o[0]
        else:
            self._out_prog = o[0]

    def before_batch(self):
        self.clear()

    def after_loss(self):
        if not self.training:
            return
        if self._out_diag is None or self._out_prog is None:
            return

        # redefine variables for convenience
        diag = self._out_diag
        prog = self._out_prog
        self.clear()
        model = self.inner_model
        win_reg = self.osp_n_skip if self.osp_n_skip is not None else model.init_sz

        diag_trunc = diag
        if diag.shape[2] > prog.shape[2]:
            diag_trunc = diag_trunc[:, :, -prog.shape[2] :]

        # sync diag prog hidden states loss
        if self.p_state_sync > 0:
            # check if diag length has to be reduced to prog length

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

            self.learn.loss_grad += self.p_state_sync * hidden_loss
            self.learn.loss += self.p_state_sync * hidden_loss

        # self.diag loss
        if self.p_diag_loss > 0:
            y_diag = model.final(diag_trunc[-1])
            if self._output_norm is not None:
                y_diag = self._output_norm.denormalize(y_diag)
            hidden_loss = self.targ_loss_func(y_diag, self.yb[0][:, -y_diag.shape[1] :])
            self.learn.loss_grad += self.p_diag_loss * hidden_loss
            self.learn.loss += self.p_diag_loss * hidden_loss

        # osp loss - one step prediction on every element of the sequence
        if self.p_osp_loss > 0 or self.p_osp_sync > 0:
            inp = self.xb[0][:, win_reg:]
            bs, n, _ = inp.shape
            # transform to a single batch of prediction length 1
            # import pdb;pdb.set_trace()
            inp = torch.flatten(inp[:, :, : model.n_u], start_dim=0, end_dim=1)[:, None, :]
            h_init = torch.flatten(diag[:, :, win_reg - 1 : -1], start_dim=1, end_dim=2)[:, None]

            out, _ = model.rnn_prognosis(inp, h_init)
            # undo transform of hiddenstates to original sequence length
            h_out = out[:, :, 0]  # the hidden state vector, 0 is the index of the single time step taken
            out = out[-1].unflatten(0, (bs, n))[
                :, :, 0
            ]  # the single step batch transformed back to the batch of sequences

            # osp hidden sync loss - deviation between diagnosis hidden state and one step prediction hidden state
            h_out_targ = torch.flatten(diag[:, :, win_reg:], start_dim=1, end_dim=2)
            hidden_loss = ((h_out_targ - h_out) / (h_out.norm() + h_out_targ.norm())).pow(2).mean()
            # import pdb;pdb.set_trace()
            self.learn.loss_grad += self.p_osp_sync * hidden_loss
            self.learn.loss += self.p_osp_sync * hidden_loss

            # osp target loss - one step prediction error on every timestep
            y_osp = model.final(out)
            if self._output_norm is not None:
                y_osp = self._output_norm.denormalize(y_osp)
            hidden_loss = self.targ_loss_func(y_osp, self.yb[0][:, -y_osp.shape[1] :])
            # import pdb;pdb.set_trace()
            self.learn.loss_grad += self.p_osp_loss * hidden_loss
            self.learn.loss += self.p_osp_loss * hidden_loss

        # tar hidden loss
        if self.p_tar_loss > 0:
            h = torch.cat([diag[:, :, : model.init_sz], prog], 2)
            h_diff = h[:, :, 1:] - h[:, :, :-1]
            hidden_loss = h_diff.pow(2).mean()

            # import pdb;pdb.set_trace()
            self.learn.loss_grad += self.p_tar_loss * hidden_loss
            self.learn.loss += self.p_tar_loss * hidden_loss


class FranSysCallback_variable_init(Callback):
    """Randomizes the diagnosis initialization window size during training.

    Args:
        init_sz_min: minimum initialization window size
        init_sz_max: maximum initialization window size (inclusive)
        model: explicit FranSys model reference, auto-detected if None
    """

    def __init__(self, init_sz_min: int, init_sz_max: int, model: nn.Module | None = None, **kwargs):
        super().__init__(**kwargs)
        self.init_sz_valid = None
        self.init_sz_min = init_sz_min
        self.init_sz_max = init_sz_max
        self.inner_model = model

    def before_fit(self):
        if self.inner_model is None:
            from ..models.layers import unwrap_model

            self.inner_model = unwrap_model(self.learn.model)

    def before_batch(self):
        if hasattr(self.inner_model, "init_sz"):
            if self.init_sz_valid is None:
                self.init_sz_valid = self.inner_model.init_sz
            if self.training:
                self.inner_model.init_sz = np.random.randint(self.init_sz_min, self.init_sz_max + 1)
            else:
                self.inner_model.init_sz = self.init_sz_valid


@delegates(FranSys, keep=True)
def FranSysLearner(
    dls: DataLoaders,
    init_sz: int,
    attach_output: bool = False,
    loss_func=nn.L1Loss(),
    metrics=fun_rmse,
    opt_func=Adam,
    lr: float = 3e-3,
    cbs: list | None = None,
    input_norm: type | None = StandardScaler1D,
    output_norm: type | None = None,
    **kwargs,
) -> Learner:
    """Create a Learner configured for FranSys diagnosis/prognosis training.

    Args:
        dls: DataLoaders with norm_stats for input/output normalization
        init_sz: number of initial time steps used for diagnosis
        attach_output: if True, use PredictionCallback to concatenate output
            to input
        loss_func: loss function, wrapped with SkipNLoss(init_sz)
        metrics: metrics to track, each wrapped with SkipNLoss(init_sz)
        opt_func: optimizer constructor
        lr: learning rate
        cbs: additional callbacks
        input_norm: scaler class for input normalization, None to disable
        output_norm: scaler class for output denormalization, None to disable
    """
    cbs = [] if cbs is None else list(cbs)
    metrics = list(metrics) if is_iter(metrics) else [metrics]

    _batch = dls.one_batch()
    inp = _batch[0].shape[-1]
    out = _batch[1].shape[-1]

    ensure_norm_stats(dls)
    norm_u, norm_x, norm_y = dls.norm_stats

    if attach_output:
        model = FranSys(inp, out, init_sz, **kwargs)

        # if PredictionCallback is not in cbs, add it
        if not any(isinstance(cb, PredictionCallback) for cb in cbs):
            cbs.append(PredictionCallback(0))

        # Input will be [u, y] after PredictionCallback concatenation
        combined_input_stats = norm_u + norm_y
    else:
        model = FranSys(inp - out, out, init_sz, **kwargs)

        # Input is [u, x?, y] from prediction-mode dls
        parts = [norm_u] + ([norm_x] if norm_x else []) + [norm_y]
        combined_input_stats = sum(parts[1:], parts[0])

    # Wrap model with input normalization and optional output denormalization
    if input_norm is not None:
        in_scaler = input_norm.from_stats(combined_input_stats)
        out_scaler = output_norm.from_stats(norm_y) if output_norm is not None else None
        model = NormalizedModel(model, in_scaler, out_scaler)

    # for long sequences, add a TruncateSequenceCallback
    seq_len = _batch[0].shape[1]
    LENGTH_THRESHOLD = 300
    if seq_len > init_sz + LENGTH_THRESHOLD:
        if not any(isinstance(cb, CB_TruncateSequence) for cb in cbs):
            INITIAL_SEQ_LEN = 100  # initial sequence length for truncation, increases during training
            cbs.append(CB_TruncateSequence(init_sz + INITIAL_SEQ_LEN))

    skip = partial(SkipNLoss, n_skip=init_sz)

    metrics = [skip(f) for f in metrics]
    loss_func = skip(loss_func)

    lrn = Learner(dls, model, loss_func=loss_func, metrics=metrics, cbs=cbs, opt_func=opt_func, lr=lr)
    return lrn
