"""FranSys prediction framework: diagnosis/prognosis models and training utilities."""

__all__ = [
    "Diag_RNN",
    "Diag_RNN_raw",
    "Diag_TCN",
    "ARProg_Init",
    "FranSys",
    "FranSysLearner",
]

import random

import torch
import torch.nn.functional as F
from torch import nn

from ..training import Learner, fun_rmse, prediction_concat, truncate_sequence
from ..models.cnn import TCN
from ..models.layers import AR_Model, SeqLinear
from ..models.scaling import ScaledModel, StandardScaler
from ..models.state import discover_state_spec, unflatten_state
from ..models.rnn import RNN, SimpleRNN


class Diag_RNN(nn.Module):
    """RNN-based diagnosis model that outputs flat state vectors.

    Outputs ``[batch, seq_len, output_size]`` — a flat state estimate per timestep.

    Args:
        input_size: number of input features
        output_size: flat state dimension (typically ``state_spec.state_size``)
        hidden_size: number of hidden units in the RNN
        rnn_layer: number of RNN layers
        linear_layer: number of linear layers in the output head
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 100,
        rnn_layer: int = 1,
        linear_layer: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.rnn = RNN(input_size, hidden_size, rnn_layer, ret_full_hidden=False, **kwargs)
        self.final = SeqLinear(hidden_size, output_size, hidden_layer=linear_layer - 1)

    def forward(self, x: torch.Tensor, init_state: list | None = None) -> tuple[torch.Tensor, list]:
        out, hidden = self.rnn(x, init_state)
        out = self.final(out)
        return out, hidden


class Diag_RNN_raw(nn.Module):
    """Raw RNN diagnosis model that returns flattened hidden states directly.

    Args:
        input_size: number of input features
        hidden_size: number of hidden units in the RNN
        rnn_layer: number of RNN layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 100,
        rnn_layer: int = 1,
    ):
        super().__init__()
        self.rnn = RNN(input_size, hidden_size, rnn_layer, ret_full_hidden=True)

    def forward(self, x: torch.Tensor, init_state: list | None = None) -> tuple[torch.Tensor, list]:
        out, hidden = self.rnn(x, init_state)
        # Flatten from [num_layers, batch, seq_len, hidden] to [batch, seq_len, state_dim]
        out = out.permute(1, 2, 0, 3).contiguous()
        out = out.view(out.shape[0], out.shape[1], -1)
        return out, hidden


class Diag_TCN(nn.Module):
    """TCN-based diagnosis model with optional MLP output head.

    Args:
        input_size: number of input features
        output_size: flat state dimension (typically ``state_spec.state_size``)
        hl_width: width of TCN hidden layers
        mlp_layers: number of additional MLP layers after the TCN
    """

    def __init__(self, input_size: int, output_size: int, hl_width: int, mlp_layers: int = 0, **kwargs):
        super().__init__()
        if mlp_layers > 0:
            self._model = TCN(input_size, hl_width, hl_width=hl_width, **kwargs)
            self.final = SeqLinear(hl_width, output_size, hidden_size=hl_width, hidden_layer=mlp_layers)
        else:
            self._model = TCN(input_size, output_size, hl_width=hl_width, **kwargs)
            self.final = nn.Identity()

    def forward(self, x: torch.Tensor, init_state: list | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._model(x)
        out = self.final(out)
        return out, out


class ARProg_Init(nn.Module):
    """Autoregressive prognosis model with diagnosis-based initialization.

    Args:
        n_u: number of input channels
        n_y: number of output channels
        init_sz: number of initial time steps used for diagnosis
        prognosis: custom AR prognosis model (default: AR_Model wrapping SimpleRNN)
        n_x: number of external state channels
        hidden_size: number of hidden units (for default prognosis and diagnosis)
        rnn_layer: number of RNN layers (for default prognosis and diagnosis)
        diag_model: custom diagnosis model outputting ``[batch, seq, state_dim]``
        linear_layer: number of linear layers in the diagnosis output head
        final_layer: number of additional final layers (unused, reserved)
        init_sz_range: if set, randomize init_sz within (min, max) during training
    """

    def __init__(
        self,
        n_u: int,
        n_y: int,
        init_sz: int,
        prognosis: nn.Module | None = None,
        n_x: int = 0,
        hidden_size: int = 100,
        rnn_layer: int = 1,
        diag_model: nn.Module | None = None,
        linear_layer: int = 1,
        final_layer: int = 0,
        init_sz_range: tuple[int, int] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.init_sz = init_sz
        self.init_sz_range = init_sz_range
        self.n_x = n_x

        if prognosis is None:
            rnn_kwargs = dict(hidden_size=hidden_size, num_layers=rnn_layer)
            rnn_kwargs = dict(rnn_kwargs, **kwargs)
            self.prognosis = AR_Model(
                SimpleRNN(input_size=n_u + n_x + n_y, output_size=n_x + n_y, return_state=True, **rnn_kwargs),
                model_has_state=True,
                return_state=False,
                ar=True,
                out_sz=n_x + n_y,
            )
        else:
            self.prognosis = prognosis

        # Discover state spec from inner RNN
        inner_rnn = self.prognosis.model.rnn
        self._state_spec = discover_state_spec(inner_rnn, n_u + n_x + n_y, device="cpu")

        if diag_model is None:
            self.diagnosis = Diag_RNN(
                n_u + n_x + n_y,
                self._state_spec.state_size,
                hidden_size=hidden_size,
                rnn_layer=rnn_layer,
                linear_layer=linear_layer,
            )
        else:
            self.diagnosis = diag_model

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        init_sz = random.randint(*self.init_sz_range) if self.training and self.init_sz_range else self.init_sz
        self._effective_init_sz = init_sz

        y_x = inp[..., self.n_u :]  # measured output and external state
        u = inp[..., : self.n_u]  # measured input

        if self.training:
            out_diag, _ = self.diagnosis(inp)
        else:
            out_diag, _ = self.diagnosis(inp[:, :init_sz])
        h_init = unflatten_state(out_diag[:, init_sz - 1], self._state_spec)
        prog_state = {
            "h": h_init,
            "y_init": y_x[:, init_sz : init_sz + 1],
        }
        out_prog = self.prognosis(u[:, init_sz:], state=prog_state, ar=True)

        result = torch.cat([torch.zeros(inp.shape[0], init_sz, y_x.shape[2], device=inp.device), out_prog], 1)

        return result[..., -self.n_y :]


class FranSys(nn.Module):
    """Framework for Analysis of Systems: combined diagnosis/prognosis model.

    Accepts any stateful prognosis model that returns ``(output, state)`` with
    output shaped ``[B, seq, features]`` (or ``[layers, B, seq, H]`` for RNNs
    with ``ret_full_hidden=True``).

    Args:
        n_u: number of input channels
        n_y: number of output channels
        init_sz: number of initial time steps used for diagnosis
        prognosis: stateful model returning ``(output, state)``
        n_x: number of external state channels
        hidden_size: hidden units for default diagnosis model
        rnn_layer: number of layers for default diagnosis model
        diag_model: custom diagnosis model outputting ``[batch, seq, state_dim]``
        linear_layer: number of linear layers in the diagnosis output head
        init_diag_only: if True, limit diagnosis to init_sz time steps during training
        final_layer: number of additional layers in the shared output head
        init_sz_range: if set, randomize init_sz within (min, max) during training
    """

    def __init__(
        self,
        n_u: int,
        n_y: int,
        init_sz: int,
        prognosis: nn.Module,
        n_x: int = 0,
        hidden_size: int = 100,
        rnn_layer: int = 1,
        diag_model: nn.Module | None = None,
        linear_layer: int = 1,
        init_diag_only: bool = False,
        final_layer: int = 0,
        init_sz_range: tuple[int, int] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x
        self.init_sz = init_sz
        self.init_diag_only = init_diag_only
        self.init_sz_range = init_sz_range

        self.prognosis = prognosis
        self._state_spec = discover_state_spec(prognosis, n_u, device="cpu")

        if diag_model is None:
            self.diagnosis = Diag_RNN(
                n_u + n_x + n_y,
                self._state_spec.state_size,
                hidden_size=hidden_size,
                rnn_layer=rnn_layer,
                linear_layer=linear_layer,
                **kwargs,
            )
        else:
            self.diagnosis = diag_model

        # Auto-discover prognosis output feature dim
        with torch.no_grad():
            output, _ = prognosis(torch.zeros(2, 1, n_u))
        out_features = output.shape[-1]
        self.final = SeqLinear(out_features, n_y, hidden_layer=final_layer)

    @staticmethod
    def _diag_output(result) -> torch.Tensor:
        """Extract flat diagnosis tensor from model output (handles tuple or tensor)."""
        return result[0] if isinstance(result, tuple) else result

    def forward(self, x: torch.Tensor, init_state: list | None = None) -> torch.Tensor:
        init_sz = random.randint(*self.init_sz_range) if self.training and self.init_sz_range else self.init_sz
        self._effective_init_sz = init_sz

        x_diag = x[..., : self.n_u + self.n_x + self.n_y]
        x_prog = x[..., : self.n_u]

        if init_state is not None:
            out_prog, _ = self.prognosis(x_prog, state=init_state)
            if out_prog.dim() > 3:
                out_prog = out_prog[-1]
            return self.final(out_prog)

        x_diag_input = x_diag if not self.init_diag_only else x_diag[:, :init_sz]
        if not self.training:
            x_diag_input = x_diag[:, :init_sz]

        out_diag = self._diag_output(self.diagnosis(x_diag_input))
        h_init = unflatten_state(out_diag[:, init_sz - 1], self._state_spec)
        out_prog, _ = self.prognosis(x_prog[:, init_sz:], state=h_init)
        if out_prog.dim() > 3:
            out_prog = out_prog[-1]
        out_prog = self.final(out_prog)
        return F.pad(out_prog, (0, 0, init_sz, 0))


def FranSysLearner(
    dls,
    init_sz: int,
    attach_output: bool = False,
    loss_func=nn.L1Loss(),
    metrics: list | None = None,
    opt_func=torch.optim.Adam,
    lr: float = 3e-3,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    input_norm: type | None = StandardScaler,
    output_norm: type | None = None,
    prognosis: nn.Module | None = None,
    hidden_size: int = 100,
    rnn_layer: int = 1,
    n_x: int = 0,
    diag_model: nn.Module | None = None,
    linear_layer: int = 1,
    init_diag_only: bool = False,
    final_layer: int = 0,
    init_sz_range: tuple[int, int] | None = None,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
) -> Learner:
    """Create a Learner configured for FranSys diagnosis/prognosis training.

    Args:
        dls: DataLoaders with norm_stats for input/output normalization
        init_sz: number of initial time steps used for diagnosis
        attach_output: if True, use prediction_concat to concatenate output to input
        loss_func: loss function
        metrics: metrics to track
        opt_func: optimizer constructor
        lr: learning rate
        transforms: additional transforms (train + valid)
        augmentations: additional augmentations (train only)
        aux_losses: additional auxiliary losses
        grad_clip: max gradient norm for clipping, or None to disable
        plot_fn: plotting function for show_batch/show_results
        input_norm: scaler class for input normalization, None to disable
        output_norm: scaler class for output denormalization, None to disable
        prognosis: custom prognosis model (default: RNN with ret_full_hidden=True)
        hidden_size: hidden units for default prognosis and diagnosis
        rnn_layer: number of RNN layers for default prognosis and diagnosis
        n_x: number of external state channels
        diag_model: custom diagnosis model
        linear_layer: number of linear layers in the diagnosis output head
        init_diag_only: if True, limit diagnosis to init_sz during training
        final_layer: number of additional layers in the output head
        init_sz_range: if set, randomize init_sz during training
        device: target device (auto-detected if None)
        show_bar: whether to show tqdm progress bars
    """
    if metrics is None:
        metrics = [fun_rmse]
    transforms = list(transforms) if transforms else []
    augmentations = list(augmentations) if augmentations else []
    aux_losses = list(aux_losses) if aux_losses else []

    _batch = dls.one_batch()
    inp = _batch[0].shape[-1]
    out = _batch[1].shape[-1]

    norm_u, norm_y = dls.norm_stats

    if attach_output:
        n_u = inp
        # Add prediction_concat transform if not already present
        if not any(isinstance(t, prediction_concat) for t in transforms):
            transforms.insert(0, prediction_concat(t_offset=0))
        combined_input_stats = norm_u + norm_y
    else:
        n_u = inp - out
        combined_input_stats = norm_u + norm_y

    if prognosis is None:
        prognosis = RNN(n_u, hidden_size=hidden_size, num_layers=rnn_layer, ret_full_hidden=True, **kwargs)

    model = FranSys(
        n_u,
        out,
        init_sz,
        prognosis,
        n_x=n_x,
        hidden_size=hidden_size,
        rnn_layer=rnn_layer,
        diag_model=diag_model,
        linear_layer=linear_layer,
        init_diag_only=init_diag_only,
        final_layer=final_layer,
        init_sz_range=init_sz_range,
        **kwargs,
    )

    # Wrap model with input normalization and optional output denormalization
    if input_norm is not None:
        in_scaler = input_norm.from_stats(combined_input_stats)
        out_scaler = output_norm.from_stats(norm_y) if output_norm is not None else None
        model = ScaledModel(model, in_scaler, out_scaler)

    # For long sequences, add truncate_sequence augmentation
    seq_len = _batch[0].shape[1]
    LENGTH_THRESHOLD = 300
    if seq_len > init_sz + LENGTH_THRESHOLD:
        if not any(isinstance(a, truncate_sequence) for a in augmentations):
            INITIAL_SEQ_LEN = 100
            augmentations.append(truncate_sequence(init_sz + INITIAL_SEQ_LEN))

    return Learner(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=init_sz,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
    )
