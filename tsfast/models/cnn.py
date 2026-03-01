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

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Mish
from torch.nn.utils.parametrizations import weight_norm

from ..training import Learner, TimeSeriesRegularizerLoss, ar_init, fun_rmse
from ..tsdata import get_io_size
from .layers import AR_Model, NormalizedModel, Scaler, SeqLinear, StandardScaler1D
from .rnn import SimpleRNN


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

    def forward(self, x: Tensor) -> Tensor:
        padding = torch.zeros((x.shape[0], x.shape[1], self.__init_size), device=x.device)
        return super().forward(torch.cat([padding, x], dim=-1))


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


class TCN_Block(nn.Module):
    """Single TCN residual block with stacked causal convolutions.

    Args:
        input_size: Number of input channels.
        output_size: Number of output channels.
        num_layers: Number of causal convolution layers in the block.
        activation: Activation function class, or None to disable.
        wn: Whether to apply weight normalization.
        bn: Whether to apply batch normalization.
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
        **kwargs,
    ):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            conv = CausalConv1d(input_size, output_size, 2, **kwargs)
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
            TCN_Block(
                input_size if i == 0 else hl_width,
                hl_width,
                dilation=2 ** (i),
                bn=bn,
                activation=act,
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


def TCNLearner(
    dls,
    num_layers: int = 3,
    hidden_size: int = 100,
    loss_func: nn.Module = nn.L1Loss(),
    metrics: list | None = None,
    n_skip: int | None = None,
    opt_func: type = torch.optim.Adam,
    input_norm: type[Scaler] | None = StandardScaler1D,
    output_norm: type[Scaler] | None = None,
    **kwargs,
) -> Learner:
    """Create a Learner with a TCN model.

    Args:
        dls: DataLoaders providing training and validation data.
        num_layers: Number of TCN hidden layers (sets receptive field to 2**num_layers).
        hidden_size: Number of channels in hidden TCN layers.
        loss_func: Loss function instance.
        metrics: List of metric functions.
        n_skip: Number of initial time steps to skip in the loss (defaults to 2**num_layers).
        opt_func: Optimizer constructor.
        input_norm: Input normalization scaler class, or None to disable.
        output_norm: Output denormalization scaler class, or None to disable.
        **kwargs: Additional arguments passed to ``TCN``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    n_skip = 2**num_layers if n_skip is None else n_skip
    model = TCN(inp, out, num_layers, hidden_size, **kwargs)
    model = NormalizedModel.from_dls(model, dls, input_norm, output_norm)

    return Learner(model, dls, loss_func=loss_func, opt_func=opt_func, metrics=metrics, n_skip=n_skip, lr=3e-3)


class SeperateTCN(nn.Module):
    """TCN with separate convolutional branches per input group, merged by a linear head.

    Args:
        input_list: List of channel counts, one per input group.
        output_size: Number of output channels.
        hl_depth: Number of TCN blocks per branch.
        hl_width: Total hidden width split evenly across branches.
        act: Activation function class.
        bn: Whether to apply batch normalization.
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

        self.final = SeqLinear(tcn_width * len(input_list), output_size, hidden_size=hl_width, hidden_layer=final_layer)

    def forward(self, x: Tensor) -> Tensor:
        tcn_out = [
            layer(x[..., self.input_list[i] : self.input_list[i + 1]].transpose(1, 2))
            for i, layer in enumerate(self.layers)
        ]
        out = torch.cat(tcn_out, dim=1).transpose(1, 2)
        return self.final(out)


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
    ):
        super().__init__()
        self.cnn = TCN(input_size, num_ft, num_cnn_layers, hs_cnn, act=nn.ReLU)
        self.rnn = SimpleRNN(
            num_ft,
            output_size,
            num_layers=num_rnn_layers,
            hidden_size=hs_rnn,
            hidden_p=hidden_p,
            input_p=input_p,
            weight_p=weight_p,
            rnn_type=rnn_type,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.rnn(self.cnn(x))


def CRNNLearner(
    dls,
    loss_func: nn.Module = nn.L1Loss(),
    metrics: list | None = None,
    n_skip: int = 0,
    opt_func: type = torch.optim.Adam,
    input_norm: type[Scaler] | None = StandardScaler1D,
    output_norm: type[Scaler] | None = None,
    **kwargs,
) -> Learner:
    """Create a Learner with a CRNN model.

    Args:
        dls: DataLoaders providing training and validation data.
        loss_func: Loss function instance.
        metrics: List of metric functions.
        n_skip: Number of initial time steps to skip in the loss.
        opt_func: Optimizer constructor.
        input_norm: Input normalization scaler class, or None to disable.
        output_norm: Output denormalization scaler class, or None to disable.
        **kwargs: Additional arguments passed to ``CRNN``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    model = CRNN(inp, out, **kwargs)
    model = NormalizedModel.from_dls(model, dls, input_norm, output_norm)

    return Learner(model, dls, loss_func=loss_func, opt_func=opt_func, metrics=metrics, n_skip=n_skip, lr=3e-3)


def AR_TCNLearner(
    dls,
    hl_depth: int = 3,
    alpha: float = 1,
    beta: float = 1,
    metrics: list | None = None,
    n_skip: int | None = None,
    opt_func: type = torch.optim.Adam,
    input_norm: type[Scaler] | None = StandardScaler1D,
    **kwargs,
) -> Learner:
    """Create a Learner with an autoregressive TCN model.

    Args:
        dls: DataLoaders providing training and validation data.
        hl_depth: Number of TCN hidden layers.
        alpha: Regularization weight for smoothness penalty.
        beta: Regularization weight for sparsity penalty.
        metrics: Metric functions (defaults to RMSE).
        n_skip: Number of initial time steps to skip in the loss (defaults to 2**hl_depth).
        opt_func: Optimizer constructor.
        input_norm: Input normalization scaler class, or None to disable.
        **kwargs: Additional arguments passed to ``TCN``.
    """
    if metrics is None:
        metrics = [fun_rmse]
    n_skip = 2**hl_depth if n_skip is None else n_skip

    inp, out = get_io_size(dls)
    ar_model = AR_Model(TCN(inp + out, out, hl_depth, **kwargs), ar=False)
    conv_module = ar_model.model.conv_layers[-1]

    model = NormalizedModel.from_dls(ar_model, dls, input_norm, autoregressive=True)

    return Learner(
        model,
        dls,
        loss_func=nn.L1Loss(),
        opt_func=opt_func,
        metrics=metrics,
        n_skip=n_skip,
        lr=3e-3,
        transforms=[ar_init()],
        aux_losses=[TimeSeriesRegularizerLoss(modules=[conv_module], alpha=alpha, beta=beta)],
    )


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
    ):
        super().__init__()
        self.cnn = SeperateTCN(input_list, num_ft, num_cnn_layers, hs_cnn, act=nn.ReLU, final_layer=0)
        self.rnn = SimpleRNN(
            num_ft,
            output_size,
            num_layers=num_rnn_layers,
            hidden_size=hs_rnn,
            hidden_p=hidden_p,
            input_p=input_p,
            weight_p=weight_p,
            rnn_type=rnn_type,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.rnn(self.cnn(x))
