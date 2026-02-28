"""FranSys prediction framework: diagnosis/prognosis models and training utilities."""

__all__ = [
    "Diag_RNN",
    "Diag_RNN_raw",
    "DiagLSTM",
    "Diag_TCN",
    "ARProg_Init",
    "FranSys",
    "FranSysLearner",
]

import random

import torch
from torch import nn

from ..training import Learner, fun_rmse, prediction_concat, truncate_sequence
from ..models.cnn import TCN
from ..models.layers import AR_Model, NormalizedModel, SeqLinear, StandardScaler1D
from ..models.rnn import RNN, SimpleRNN


class Diag_RNN(nn.Module):
    """RNN-based diagnosis model with configurable output layers.

    Args:
        input_size: number of input features
        output_size: number of output features per output layer
        output_layer: number of stacked output layers
        hidden_size: number of hidden units in the RNN
        rnn_layer: number of RNN layers
        linear_layer: number of linear layers in the output head
    """

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
        self.output_size = output_size

        self.rnn = RNN(input_size, hidden_size, rnn_layer, ret_full_hidden=False, **kwargs)
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


class Diag_RNN_raw(nn.Module):
    """Raw RNN diagnosis model that returns full hidden states directly.

    Args:
        input_size: number of input features
        output_size: number of output features
        output_layer: number of stacked output layers
        hidden_size: number of hidden units in the RNN
        rnn_layer: number of RNN layers
        linear_layer: number of linear layers (unused, kept for API compatibility)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_layer: int = 1,
        hidden_size: int = 100,
        rnn_layer: int = 1,
        linear_layer: int = 1,
    ):
        super().__init__()

        self.rnn = RNN(input_size, output_size, output_layer, ret_full_hidden=True)

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


class Diag_TCN(nn.Module):
    """TCN-based diagnosis model with optional MLP output head.

    Args:
        input_size: number of input features
        output_size: number of output features per output layer
        output_layer: number of stacked output layers
        hl_width: width of TCN hidden layers
        mlp_layers: number of additional MLP layers after the TCN
    """

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
        init_sz_range: if set, randomize init_sz within (min, max) during training
    """

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
        init_sz_range: tuple[int, int] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.init_sz = init_sz
        self.init_sz_range = init_sz_range
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
        init_sz = random.randint(*self.init_sz_range) if self.training and self.init_sz_range else self.init_sz
        self._effective_init_sz = init_sz

        y_x = inp[..., self.n_u :]  # measured output and external state
        u = inp[..., : self.n_u]  # measured input

        if self.training:
            out_diag, _ = self.rnn_diagnosis(inp)
        else:
            out_diag, _ = self.rnn_diagnosis(inp[:, :init_sz])
        h_init = self.rnn_diagnosis.output_to_hidden(out_diag, init_sz - 1)
        prog_state = {
            "h": h_init,
            "y_init": y_x[:, init_sz : init_sz + 1],
        }
        out_prog = self.rnn_prognosis(u[:, init_sz:], state=prog_state, ar=True)

        result = torch.cat([torch.zeros(inp.shape[0], init_sz, y_x.shape[2], device=inp.device), out_prog], 1)

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
        init_sz_range: if set, randomize init_sz within (min, max) during training
    """

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
        init_sz = random.randint(*self.init_sz_range) if self.training and self.init_sz_range else self.init_sz
        self._effective_init_sz = init_sz

        x_diag = x[..., : self.n_u + self.n_x + self.n_y]
        x_prog = x[..., : self.n_u]

        if self.init_diag_only:
            x_diag = x_diag[:, :init_sz]  # limit diagnosis length to init size

        if self.training:
            # in training, estimate the full sequence with the diagnosis module
            if init_state is None:
                # execution with no initial state
                out_diag, _ = self.rnn_diagnosis(x_diag)
                h_init = self.rnn_diagnosis.output_to_hidden(out_diag, init_sz - 1)

                # ToDo: only execute this if callback is used
                new_hidden = self.rnn_diagnosis.output_to_hidden(out_diag, -1)

                out_prog, _ = self.rnn_prognosis(x_prog[:, init_sz:].contiguous(), h_init)
                out_prog = torch.cat([out_diag[:, :, :init_sz], out_prog], 2)
            else:
                out_prog, _ = self.rnn_prognosis(x_prog, init_state)
                out_diag, _ = self.rnn_diagnosis(x_diag)
                new_hidden = self.rnn_diagnosis.output_to_hidden(out_diag, -1)
        else:
            # in inference, use the diagnosis module only for initial state estimation
            if init_state is None:
                out_init, _ = self.rnn_diagnosis(x_diag[:, :init_sz])
                h_init = self.rnn_diagnosis.output_to_hidden(out_init, -1)
                out_prog, new_hidden = self.rnn_prognosis(x_prog[:, init_sz:], h_init)
                out_prog = torch.cat([out_init, out_prog], 2)
            else:
                out_prog, new_hidden = self.rnn_prognosis(x_prog, init_state)

        # Shared Linear Layer
        result = self.final(out_prog[-1])
        return result


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
    input_norm: type | None = StandardScaler1D,
    output_norm: type | None = None,
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
        input_norm: scaler class for input normalization, None to disable
        output_norm: scaler class for output denormalization, None to disable
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
        model = FranSys(inp, out, init_sz, **kwargs)

        # Add prediction_concat transform if not already present
        if not any(isinstance(t, prediction_concat) for t in transforms):
            transforms.insert(0, prediction_concat(t_offset=0))

        # Input will be [u, y] after prediction_concat
        combined_input_stats = norm_u + norm_y
    else:
        model = FranSys(inp - out, out, init_sz, **kwargs)

        # Input is [u, y] from prediction-mode dls
        combined_input_stats = norm_u + norm_y

    # Wrap model with input normalization and optional output denormalization
    if input_norm is not None:
        in_scaler = input_norm.from_stats(combined_input_stats)
        out_scaler = output_norm.from_stats(norm_y) if output_norm is not None else None
        model = NormalizedModel(model, in_scaler, out_scaler)

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
        n_skip=init_sz,
        opt_func=opt_func,
        lr=lr,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
    )
