"""Convolutional and temporal convolutional network architectures for time series."""

__all__ = [
    "Conv1D",
    "CNN",
    "CausalConv1d",
    "CConv1D",
    "TCN_Block",
    "TCN",
    "TCNLearner",
    "SeperateTCN",
    "CRNN",
    "CRNNLearner",
    "AR_TCNLearner",
    "SeperateCRNN",
]

from functools import partial

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Mish
from torch.nn.utils.parametrizations import weight_norm

from fastcore.meta import delegates
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.optimizer import Adam
from fastai.torch_basics import to_detach

from ..data.loader import get_inp_out_size
from ..learner.callbacks import ARInitCB, TimeSeriesRegularizer
from ..learner.losses import SkipNLoss, fun_rmse
from .layers import AR_Model, NormalizedModel, Scaler, SeqLinear, StandardScaler1D
from .rnn import SimpleRNN


@delegates(nn.Conv1d, keep=True)
def Conv1D(
    input_size: int,
    output_size: int,
    kernel_size: int = 3,
    activation: type[nn.Module] | None = Mish,
    wn: bool = True,
    bn: bool = False,
    **kwargs,
) -> nn.Sequential:
    """Create a 1D convolutional block with optional activation and batch norm.

    Args:
        input_size: Number of input channels.
        output_size: Number of output channels.
        kernel_size: Size of the convolving kernel.
        activation: Activation function class, or None to disable.
        wn: Whether to apply weight normalization.
        bn: Whether to apply batch normalization before the convolution.
        **kwargs: Additional arguments passed to ``nn.Conv1d``.
    """
    conv = nn.Conv1d(input_size, output_size, kernel_size, **kwargs)
    act = activation() if activation is not None else None
    bn = nn.BatchNorm1d(input_size) if bn else None
    m = [m for m in [bn, conv, act] if m is not None]
    return nn.Sequential(*m)


class CNN(nn.Module):
    """Simple stacked 1D CNN with a pointwise output projection.

    Args:
        input_size: Number of input channels.
        output_size: Number of output channels.
        hl_depth: Number of hidden convolutional layers.
        hl_width: Number of channels in each hidden layer.
        act: Activation function class.
        bn: Whether to apply batch normalization.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hl_depth: int = 1,
        hl_width: int = 10,
        act: type[nn.Module] = Mish,
        bn: bool = False,
    ):
        super().__init__()

        conv_layers = [
            Conv1D(input_size if i == 0 else hl_width, hl_width, bn=bn, activation=act, padding=1)
            for i in range(hl_depth)
        ]
        self.conv_layers = nn.Sequential(*conv_layers)
        self.final = nn.Conv1d(hl_width, output_size, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x_in = x.transpose(1, 2)
        out = self.conv_layers(x_in)
        out = self.final(out).transpose(1, 2)
        return out


class CausalConv1d(torch.nn.Conv1d):
    """Causal 1D convolution that pads only on the left to prevent future leakage.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dilation: Dilation factor for the kernel.
        groups: Number of blocked connections from input to output channels.
        bias: Whether to add a learnable bias.
        stateful: Whether to carry internal state across forward calls.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        stateful: bool = False,
    ):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.__init_size = (kernel_size - 1) * dilation
        self.x_init = None
        self.stateful = stateful

    def forward(self, x: Tensor) -> Tensor:
        if self.x_init is not None and self.x_init.shape[0] != x.shape[0]:
            self.x_init = None

        if self.x_init is None or not self.stateful:
            self.x_init = torch.zeros((x.shape[0], x.shape[1], self.__init_size), device=x.device)

        x = torch.cat([self.x_init, x], dim=-1)

        out = super().forward(x)

        if self.stateful:
            self.x_init = to_detach(x[..., -self.__init_size :], cpu=False, gather=False)

        return out

    def reset_state(self) -> None:
        self.x_init = None


@delegates(CausalConv1d, keep=True)
def CConv1D(
    input_size: int,
    output_size: int,
    kernel_size: int = 2,
    activation: type[nn.Module] | None = Mish,
    wn: bool = True,
    bn: bool = False,
    **kwargs,
) -> nn.Sequential:
    """Create a causal 1D convolutional block with optional weight norm and batch norm.

    Args:
        input_size: Number of input channels.
        output_size: Number of output channels.
        kernel_size: Size of the convolving kernel.
        activation: Activation function class, or None to disable.
        wn: Whether to apply weight normalization.
        bn: Whether to apply batch normalization before the convolution.
        **kwargs: Additional arguments passed to ``CausalConv1d``.
    """
    conv = CausalConv1d(input_size, output_size, kernel_size, **kwargs)
    if wn:
        conv = weight_norm(conv)
    act = activation() if activation is not None else None
    bn = nn.BatchNorm1d(input_size) if bn else None
    m = [m for m in [bn, conv, act] if m is not None]
    return nn.Sequential(*m)


@delegates(CausalConv1d, keep=True)
class TCN_Block(nn.Module):
    """Single TCN residual block with stacked causal convolutions.

    Args:
        input_size: Number of input channels.
        output_size: Number of output channels.
        num_layers: Number of causal convolution layers in the block.
        activation: Activation function class, or None to disable.
        wn: Whether to apply weight normalization.
        bn: Whether to apply batch normalization.
        stateful: Whether causal convolutions carry state across calls.
        **kwargs: Additional arguments passed to ``CausalConv1d``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 1,
        activation: type[nn.Module] | None = Mish,
        wn: bool = True,
        bn: bool = False,
        stateful: bool = False,
        **kwargs,
    ):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            conv = CausalConv1d(input_size, output_size, 2, stateful=stateful, **kwargs)
            if wn:
                conv = weight_norm(conv)
            act = activation() if activation is not None else None
            bn = nn.BatchNorm1d(input_size) if bn else None
            layers += [m for m in [bn, conv, act] if m is not None]

        self.layers = nn.Sequential(*layers)

        self.residual = nn.Conv1d(input_size, output_size, kernel_size=1) if output_size != input_size else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        out = out + (x if self.residual is None else self.residual(x))
        return out


class TCN(nn.Module):
    """Temporal Convolutional Network with exponentially increasing dilation.

    Args:
        input_size: Number of input channels.
        output_size: Number of output channels.
        hl_depth: Number of TCN blocks (each with doubled dilation).
        hl_width: Number of channels in each hidden TCN block.
        act: Activation function class.
        bn: Whether to apply batch normalization.
        stateful: Whether causal convolutions carry state across calls.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hl_depth: int = 1,
        hl_width: int = 10,
        act: type[nn.Module] = Mish,
        bn: bool = False,
        stateful: bool = False,
    ):
        super().__init__()

        conv_layers = [
            TCN_Block(
                input_size if i == 0 else hl_width,
                hl_width,
                dilation=2 ** (i),
                bn=bn,
                activation=act,
                stateful=stateful,
            )
            for i in range(hl_depth)
        ]
        self.conv_layers = nn.Sequential(*conv_layers)
        self.final = nn.Conv1d(hl_width, output_size, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x_in = x.transpose(1, 2)
        out = self.conv_layers(x_in)
        out = self.final(out).transpose(1, 2)
        return out


@delegates(TCN, keep=True)
def TCNLearner(
    dls: DataLoaders,
    num_layers: int = 3,
    hidden_size: int = 100,
    loss_func: nn.Module = nn.L1Loss(),
    metrics: list = [fun_rmse],
    n_skip: int | None = None,
    opt_func: type = Adam,
    cbs: list | None = None,
    input_norm: type[Scaler] | None = StandardScaler1D,
    output_norm: type[Scaler] | None = None,
    **kwargs,
) -> Learner:
    """Create a fastai Learner with a TCN model.

    Args:
        dls: DataLoaders providing training and validation data.
        num_layers: Number of TCN hidden layers (sets receptive field to 2**num_layers).
        hidden_size: Number of channels in hidden TCN layers.
        loss_func: Loss function instance.
        metrics: List of metric functions.
        n_skip: Number of initial time steps to skip in the loss (defaults to 2**num_layers).
        opt_func: Optimizer constructor.
        cbs: Additional callbacks.
        input_norm: Input normalization scaler class, or None to disable.
        output_norm: Output denormalization scaler class, or None to disable.
        **kwargs: Additional arguments passed to ``TCN``.
    """
    inp, out = get_inp_out_size(dls)
    n_skip = 2**num_layers if n_skip is None else n_skip
    model = TCN(inp, out, num_layers, hidden_size, **kwargs)
    model = NormalizedModel.from_dls(model, dls, input_norm, output_norm)

    skip = partial(SkipNLoss, n_skip=n_skip)

    metrics = [skip(f) for f in metrics]
    loss_func = skip(loss_func)

    lrn = Learner(dls, model, loss_func=loss_func, opt_func=opt_func, metrics=metrics, cbs=cbs, lr=3e-3)
    return lrn


class SeperateTCN(nn.Module):
    """TCN with separate convolutional branches per input group, merged by a linear head.

    Args:
        input_list: List of channel counts, one per input group.
        output_size: Number of output channels.
        hl_depth: Number of TCN blocks per branch.
        hl_width: Total hidden width split evenly across branches.
        act: Activation function class.
        bn: Whether to apply batch normalization.
        stateful: Whether to carry state across forward calls.
        final_layer: Number of hidden layers in the final linear head.
    """

    def __init__(
        self,
        input_list: list[int],
        output_size: int,
        hl_depth: int = 1,
        hl_width: int = 10,
        act: type[nn.Module] = Mish,
        bn: bool = False,
        stateful: bool = False,
        final_layer: int = 3,
    ):
        super().__init__()
        self.input_list = np.cumsum([0] + input_list)

        tcn_width = hl_width // len(input_list)
        layers = [
            [
                TCN_Block(n if i == 0 else tcn_width, tcn_width, dilation=2 ** (i), bn=bn, activation=act)
                for i in range(hl_depth)
            ]
            for n in input_list
        ]
        self.layers = nn.ModuleList([nn.Sequential(*layer) for layer in layers])

        self.rec_field = (2**hl_depth) - 1
        self.final = SeqLinear(tcn_width * len(input_list), output_size, hidden_size=hl_width, hidden_layer=final_layer)
        self.x_init = None
        self.stateful = stateful

    def forward(self, x: Tensor) -> Tensor:
        if self.x_init is not None:
            if self.x_init.shape[0] != x.shape[0]:
                self.x_init = None
            elif self.stateful:
                x = torch.cat([self.x_init, x], dim=1)

        tcn_out = [
            layer(x[..., self.input_list[i] : self.input_list[i + 1]].transpose(1, 2))
            for i, layer in enumerate(self.layers)
        ]
        out = torch.cat(tcn_out, dim=1).transpose(1, 2)

        out = self.final(out)

        if self.stateful:
            if self.x_init is not None:
                out = out[:, self.rec_field :]
            self.x_init = x[:, -self.rec_field :]

        return out

    def reset_state(self) -> None:
        self.x_init = None


class CRNN(nn.Module):
    """Convolutional-Recurrent Neural Network combining a TCN front-end with an RNN back-end.

    Args:
        input_size: Number of input channels.
        output_size: Number of output channels.
        num_ft: Number of intermediate features between the CNN and RNN stages.
        num_cnn_layers: Number of TCN blocks in the convolutional stage.
        num_rnn_layers: Number of stacked RNN layers.
        hs_cnn: Hidden channel width of the TCN stage.
        hs_rnn: Hidden size of the RNN stage.
        hidden_p: Dropout probability on RNN hidden-to-hidden connections.
        input_p: Dropout probability on RNN inputs.
        weight_p: Weight dropout probability for RNN parameters.
        rnn_type: RNN cell type (``"gru"`` or ``"lstm"``).
        stateful: Whether both CNN and RNN carry state across calls.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_ft: int = 10,
        num_cnn_layers: int = 4,
        num_rnn_layers: int = 2,
        hs_cnn: int = 10,
        hs_rnn: int = 10,
        hidden_p: float = 0,
        input_p: float = 0,
        weight_p: float = 0,
        rnn_type: str = "gru",
        stateful: bool = False,
    ):
        super().__init__()
        self.cnn = TCN(input_size, num_ft, num_cnn_layers, hs_cnn, act=nn.ReLU, stateful=stateful)
        self.rnn = SimpleRNN(
            num_ft,
            output_size,
            num_layers=num_rnn_layers,
            hidden_size=hs_rnn,
            hidden_p=hidden_p,
            input_p=input_p,
            weight_p=weight_p,
            rnn_type=rnn_type,
            stateful=stateful,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.rnn(self.cnn(x))


@delegates(CRNN, keep=True)
def CRNNLearner(
    dls: DataLoaders,
    loss_func: nn.Module = nn.L1Loss(),
    metrics: list = [fun_rmse],
    n_skip: int = 0,
    opt_func: type = Adam,
    cbs: list | None = None,
    input_norm: type[Scaler] | None = StandardScaler1D,
    output_norm: type[Scaler] | None = None,
    **kwargs,
) -> Learner:
    """Create a fastai Learner with a CRNN model.

    Args:
        dls: DataLoaders providing training and validation data.
        loss_func: Loss function instance.
        metrics: List of metric functions.
        n_skip: Number of initial time steps to skip in the loss.
        opt_func: Optimizer constructor.
        cbs: Additional callbacks.
        input_norm: Input normalization scaler class, or None to disable.
        output_norm: Output denormalization scaler class, or None to disable.
        **kwargs: Additional arguments passed to ``CRNN``.
    """
    inp, out = get_inp_out_size(dls)
    model = CRNN(inp, out, **kwargs)
    model = NormalizedModel.from_dls(model, dls, input_norm, output_norm)

    skip = partial(SkipNLoss, n_skip=n_skip)

    metrics = [skip(f) for f in metrics]
    loss_func = skip(loss_func)

    lrn = Learner(dls, model, loss_func=loss_func, opt_func=opt_func, metrics=metrics, cbs=cbs, lr=3e-3)
    return lrn


@delegates(TCN, keep=True)
def AR_TCNLearner(
    dls: DataLoaders,
    hl_depth: int = 3,
    alpha: float = 1,
    beta: float = 1,
    early_stop: int = 0,
    metrics: list | None = None,
    n_skip: int | None = None,
    opt_func: type = Adam,
    input_norm: type[Scaler] | None = StandardScaler1D,
    **kwargs,
) -> Learner:
    """Create a fastai Learner with an autoregressive TCN model.

    Args:
        dls: DataLoaders providing training and validation data.
        hl_depth: Number of TCN hidden layers.
        alpha: Regularization weight for smoothness penalty.
        beta: Regularization weight for sparsity penalty.
        early_stop: Early stopping patience in epochs (0 disables).
        metrics: Metric functions (defaults to RMSE with skip).
        n_skip: Number of initial time steps to skip in the loss (defaults to 2**hl_depth).
        opt_func: Optimizer constructor.
        input_norm: Input normalization scaler class, or None to disable.
        **kwargs: Additional arguments passed to ``TCN``.
    """
    n_skip = 2**hl_depth if n_skip is None else n_skip

    inp, out = get_inp_out_size(dls)
    ar_model = AR_Model(TCN(inp + out, out, hl_depth, **kwargs), ar=False)
    conv_module = ar_model.model.conv_layers[-1]

    model = NormalizedModel.from_dls(ar_model, dls, input_norm, autoregressive=True)

    cbs = [
        ARInitCB(),
        TimeSeriesRegularizer(alpha=alpha, beta=beta, modules=[conv_module]),
    ]  # SaveModelCallback()
    if early_stop > 0:
        cbs += [EarlyStoppingCallback(patience=early_stop)]

    if metrics is None:
        metrics = SkipNLoss(fun_rmse, n_skip)

    lrn = Learner(dls, model, loss_func=nn.L1Loss(), opt_func=opt_func, metrics=metrics, cbs=cbs, lr=3e-3)
    return lrn


class SeperateCRNN(nn.Module):
    """CRNN with separate TCN branches per input group, merged before the RNN stage.

    Args:
        input_list: List of channel counts, one per input group.
        output_size: Number of output channels.
        num_ft: Number of intermediate features between the CNN and RNN stages.
        num_cnn_layers: Number of TCN blocks per branch in the convolutional stage.
        num_rnn_layers: Number of stacked RNN layers.
        hs_cnn: Hidden channel width of the TCN branches.
        hs_rnn: Hidden size of the RNN stage.
        hidden_p: Dropout probability on RNN hidden-to-hidden connections.
        input_p: Dropout probability on RNN inputs.
        weight_p: Weight dropout probability for RNN parameters.
        rnn_type: RNN cell type (``"gru"`` or ``"lstm"``).
        stateful: Whether both CNN and RNN carry state across calls.
    """

    def __init__(
        self,
        input_list: list[int],
        output_size: int,
        num_ft: int = 10,
        num_cnn_layers: int = 4,
        num_rnn_layers: int = 2,
        hs_cnn: int = 10,
        hs_rnn: int = 10,
        hidden_p: float = 0,
        input_p: float = 0,
        weight_p: float = 0,
        rnn_type: str = "gru",
        stateful: bool = False,
    ):
        super().__init__()
        self.cnn = SeperateTCN(
            input_list, num_ft, num_cnn_layers, hs_cnn, act=nn.ReLU, stateful=stateful, final_layer=0
        )
        self.rnn = SimpleRNN(
            num_ft,
            output_size,
            num_layers=num_rnn_layers,
            hidden_size=hs_rnn,
            hidden_p=hidden_p,
            input_p=input_p,
            weight_p=weight_p,
            rnn_type=rnn_type,
            stateful=stateful,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.rnn(self.cnn(x))
