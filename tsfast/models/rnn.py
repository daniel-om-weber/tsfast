"""RNN-based models for time series with regularization and DenseNet variants."""

__all__ = [
    "RNN",
    "SimpleRNN",
    "RNNLearner",
    "AR_RNNLearner",
    "ResidualBlock_RNN",
    "SimpleResidualRNN",
    "DenseLayer_RNN",
    "DenseBlock_RNN",
    "DenseNet_RNN",
    "SeperateRNN",
]

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..training import Learner, TbpttLearner, TimeSeriesRegularizerLoss, ar_init, fun_rmse
from .layers import AR_Model, BatchNorm_1D_Stateful, NormalizedModel, SeqLinear, StandardScaler1D


def _dropout_mask(x: Tensor, sz: list, p: float) -> Tensor:
    """Return a multiplicative dropout mask with probability ``p`` to zero an element."""
    return x.new_empty(*sz).bernoulli_(1 - p).div_(1 - p)


class RNNDropout(nn.Module):
    """Dropout with probability ``p`` that is consistent on the seq_len dimension (variational dropout)."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        return x * _dropout_mask(x.data, (x.size(0), 1, *x.shape[2:]), self.p)


class WeightDropout(nn.Module):
    """A module that wraps another layer in which some weights will be replaced by 0 during training.

    Args:
        module: wrapped RNN module.
        weight_p: weight dropout probability.
        layer_names: name(s) of the parameters to apply dropout to.
    """

    def __init__(
        self,
        module: nn.Module,
        weight_p: float,
        layer_names: str | list[str] = "weight_hh_l0",
    ):
        super().__init__()
        self.module = module
        self.weight_p = weight_p
        self.layer_names = [layer_names] if isinstance(layer_names, str) else list(layer_names)
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            delattr(self.module, layer)
            self.register_parameter(f"{layer}_raw", nn.Parameter(w.data))
            setattr(self.module, layer, w.clone())
            if isinstance(self.module, (nn.RNNBase, nn.modules.rnn.RNNBase)):
                self.module.flatten_parameters = self._do_nothing

    def _setweights(self):
        """Apply dropout to the raw weights."""
        for layer in self.layer_names:
            raw_w = getattr(self, f"{layer}_raw")
            if self.training:
                w = F.dropout(raw_w, p=self.weight_p)
            else:
                w = raw_w.clone()
            setattr(self.module, layer, w)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore", category=UserWarning)
            return self.module(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f"{layer}_raw")
            setattr(self.module, layer, raw_w.clone())
        if hasattr(self.module, "reset"):
            self.module.reset()

    def _do_nothing(self):
        pass


class RNN(nn.Module):
    """Multi-layer RNN with dropout and normalization, inspired by https://arxiv.org/abs/1708.02182.

    Args:
        input_size: number of input features per timestep.
        hidden_size: number of hidden units per layer.
        num_layers: number of stacked RNN layers.
        hidden_p: dropout probability applied between hidden layers.
        input_p: dropout probability applied to the input.
        weight_p: weight dropout probability applied within each RNN cell.
        rnn_type: recurrent cell type, one of ``'gru'``, ``'lstm'``, or ``'rnn'``.
        ret_full_hidden: if True, return stacked hidden outputs from all layers.
        stateful: if True, enable per-timestep batch normalization layers.
        normalization: normalization between layers (``''``, ``'layernorm'``, or ``'batchnorm'``).
        **kwargs: additional keyword arguments forwarded to the underlying ``nn.RNN``/``nn.GRU``/``nn.LSTM``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        hidden_p: float = 0.0,
        input_p: float = 0.0,
        weight_p: float = 0.0,
        rnn_type: str = "gru",
        ret_full_hidden: bool = False,
        stateful: bool = False,
        normalization: str = "",
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_p = hidden_p
        self.input_p = input_p
        self.weight_p = weight_p
        self.rnn_type = rnn_type
        self.ret_full_hidden = ret_full_hidden
        self.normalization = normalization

        self.rnns = nn.ModuleList(
            [
                self._one_rnn(input_size if i == 0 else hidden_size, hidden_size, weight_p, rnn_type, **kwargs)
                for i in range(num_layers)
            ]
        )

        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for _ in range(num_layers)])

        if normalization == "":
            self.norm_layers = [None] * num_layers
        elif normalization == "layernorm":
            self.norm_layers = nn.ModuleList(
                [nn.LayerNorm(hidden_size, elementwise_affine=False) for _ in range(num_layers)]
            )
        elif normalization == "batchnorm":
            self.norm_layers = nn.ModuleList(
                [
                    (BatchNorm_1D_Stateful(hidden_size, stateful=stateful, batch_first=True, affine=False))
                    for i in range(num_layers)
                ]
            )
        else:
            raise ValueError("Invalid value for normalization")

    def forward(self, inp: Tensor, state: list | None = None):
        bs, seq_len, _ = inp.shape
        r_input = self.input_dp(inp) if self.input_p > 0 else inp
        full_hid, new_hidden = [], []
        for layer_idx, (rnn, hid_dp, nrm) in enumerate(zip(self.rnns, self.hidden_dps, self.norm_layers)):
            r_output, h = rnn(
                r_input.contiguous(),
                state[layer_idx] if state is not None else None,
            )

            if self.normalization != "":
                r_output = nrm(r_output)

            if layer_idx != self.num_layers - 1:
                r_output = hid_dp(r_output)

            full_hid.append(r_output)
            new_hidden.append(h)
            r_input = r_output

        output = r_output if not self.ret_full_hidden else torch.stack(full_hid, 0)

        return output, new_hidden

    def _one_rnn(self, n_in, n_out, weight_p, rnn_type, **kwargs):
        if rnn_type == "gru":
            rnn = nn.GRU(n_in, n_out, 1, batch_first=True, **kwargs)
            if weight_p > 0:
                rnn = WeightDropout(rnn, weight_p)
        elif rnn_type == "lstm":
            rnn = nn.LSTM(n_in, n_out, 1, batch_first=True, **kwargs)
            if weight_p > 0:
                rnn = WeightDropout(rnn, weight_p)
        elif rnn_type == "rnn":
            rnn = nn.RNN(n_in, n_out, 1, batch_first=True, **kwargs)
            if weight_p > 0:
                rnn = WeightDropout(rnn, weight_p)
        else:
            raise Exception
        return rnn


class Sequential_RNN(RNN):
    """RNN variant that returns only the output tensor, discarding hidden state."""

    def forward(self, inp: Tensor, state: list | None = None):
        return super().forward(inp, state)[0]


class SimpleRNN(nn.Module):
    """Simple RNN with a linear output head.

    Args:
        input_size: number of input features per timestep.
        output_size: number of output features per timestep.
        num_layers: number of stacked RNN layers.
        hidden_size: number of hidden units per RNN layer.
        linear_layers: number of hidden linear layers in the output head.
        return_state: if True, return ``(output, hidden_state)`` instead of just output.
        **kwargs: additional keyword arguments forwarded to ``RNN``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 1,
        hidden_size: int = 100,
        linear_layers: int = 0,
        return_state: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.return_state = return_state
        self.rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, **kwargs)
        self.final = SeqLinear(
            hidden_size, output_size, hidden_size=hidden_size, hidden_layer=linear_layers, act=nn.LeakyReLU
        )

    def forward(self, x: Tensor, state: list | None = None):
        out, h = self.rnn(x, state)
        out = self.final(out)
        return out if not self.return_state else (out, h)


def _get_inp_out_size(dls):
    """Get input/output sizes from a DataLoaders batch."""
    batch = dls.one_batch()
    return batch[0].shape[-1], batch[1].shape[-1]


def RNNLearner(
    dls,
    loss_func=nn.L1Loss(),
    metrics: list | None = None,
    n_skip: int = 0,
    num_layers: int = 1,
    hidden_size: int = 100,
    stateful: bool = False,
    sub_seq_len: int | None = None,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler1D,
    output_norm: type | None = None,
    **kwargs,
):
    """Create a Learner with a SimpleRNN model and standard training setup.

    Args:
        dls: DataLoaders providing training and validation data.
        loss_func: loss function for training.
        metrics: list of metric functions evaluated during validation.
        n_skip: number of initial timesteps to skip in loss and metric computation.
        num_layers: number of stacked RNN layers.
        hidden_size: number of hidden units per RNN layer.
        stateful: if True, enable stateful training with TBPTT.
        sub_seq_len: sub-sequence length for TBPTT (defaults to 100).
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        output_norm: scaler class for output denormalization, or None to disable.
        **kwargs: additional keyword arguments forwarded to ``SimpleRNN``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = _get_inp_out_size(dls)
    model = SimpleRNN(inp, out, num_layers, hidden_size, stateful=stateful, **kwargs)
    model = NormalizedModel.from_dls(model, dls, input_norm, output_norm)

    cls = TbpttLearner if stateful else Learner
    extra = {"sub_seq_len": sub_seq_len or 100} if stateful else {}
    return cls(model, dls, loss_func=loss_func, metrics=metrics, n_skip=n_skip, opt_func=opt_func, lr=3e-3, **extra)


def AR_RNNLearner(
    dls,
    alpha: float = 0,
    beta: float = 0,
    metrics: list | None = None,
    n_skip: int = 0,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler1D,
    **kwargs,
):
    """Create a Learner with an autoregressive RNN model.

    Args:
        dls: DataLoaders providing training and validation data.
        alpha: activation regularization penalty weight.
        beta: temporal activation regularization penalty weight.
        metrics: metric functions for validation, or None for default RMSE.
        n_skip: number of initial timesteps to skip in metric computation.
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        **kwargs: additional keyword arguments forwarded to ``SimpleRNN``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = _get_inp_out_size(dls)
    ar_model = AR_Model(SimpleRNN(inp + out, out, **kwargs), ar=False)
    rnn_module = ar_model.model.rnn

    model = NormalizedModel.from_dls(ar_model, dls, input_norm, autoregressive=True)

    return Learner(
        model,
        dls,
        loss_func=nn.L1Loss(),
        metrics=metrics,
        n_skip=n_skip,
        opt_func=opt_func,
        lr=3e-3,
        transforms=[ar_init()],
        aux_losses=[TimeSeriesRegularizerLoss(modules=[rnn_module], alpha=alpha, beta=beta)],
    )


class ResidualBlock_RNN(nn.Module):
    """Two-layer RNN block with a residual skip connection.

    Args:
        input_size: number of input features per timestep.
        hidden_size: number of hidden units in each RNN layer.
        **kwargs: additional keyword arguments forwarded to ``RNN``.
    """

    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.rnn1 = RNN(input_size, hidden_size, num_layers=1, **kwargs)
        self.rnn2 = RNN(hidden_size, hidden_size, num_layers=1, **kwargs)
        self.residual = (
            SeqLinear(input_size, hidden_size, hidden_layer=0) if hidden_size != input_size else nn.Identity()
        )

    def forward(self, x: Tensor):
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        return out + self.residual(x)


class SimpleResidualRNN(nn.Sequential):
    """Sequential stack of residual RNN blocks with a linear output head.

    Args:
        input_size: number of input features per timestep.
        output_size: number of output features per timestep.
        num_blocks: number of stacked residual RNN blocks.
        hidden_size: number of hidden units per block.
        **kwargs: additional keyword arguments forwarded to ``ResidualBlock_RNN``.
    """

    def __init__(self, input_size: int, output_size: int, num_blocks: int = 1, hidden_size: int = 100, **kwargs):
        super().__init__()
        for i in range(num_blocks):
            self.add_module(
                "rnn%d" % i, ResidualBlock_RNN(input_size if i == 0 else hidden_size, hidden_size, **kwargs)
            )

        self.add_module("linear", SeqLinear(hidden_size, output_size, hidden_size, hidden_layer=1))


class DenseLayer_RNN(nn.Module):
    """Two-layer RNN that concatenates its output with the input (DenseNet-style).

    Args:
        input_size: number of input features per timestep.
        hidden_size: growth rate (number of new features produced).
        **kwargs: additional keyword arguments forwarded to ``RNN``.
    """

    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.rnn1 = RNN(input_size, hidden_size, num_layers=1, **kwargs)
        self.rnn2 = RNN(hidden_size, hidden_size, num_layers=1, **kwargs)

    def forward(self, x: Tensor):
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        return torch.cat([x, out], 2)


class DenseBlock_RNN(nn.Sequential):
    """Sequential block of DenseNet-style RNN layers with feature concatenation.

    Args:
        num_layers: number of dense RNN layers in this block.
        num_input_features: number of input features entering the block.
        growth_rate: number of new features each dense layer adds.
        **kwargs: additional keyword arguments forwarded to ``DenseLayer_RNN``.
    """

    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int, **kwargs):
        super().__init__()
        for i in range(num_layers):
            self.add_module(
                "denselayer%d" % i, DenseLayer_RNN(num_input_features + i * growth_rate, growth_rate, **kwargs)
            )


class DenseNet_RNN(nn.Sequential):
    """DenseNet architecture using RNN layers with transition layers between blocks.

    Args:
        input_size: number of input features per timestep.
        output_size: number of output features per timestep.
        growth_rate: number of new features each dense layer adds.
        block_config: tuple specifying the number of layers in each dense block.
        num_init_features: number of features produced by the initial RNN layer.
        **kwargs: additional keyword arguments forwarded to ``RNN``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        growth_rate: int = 32,
        block_config: tuple = (3, 3),
        num_init_features: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.add_module("rnn0", Sequential_RNN(input_size, num_init_features, 1, **kwargs))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.add_module(
                "denseblock%d" % i, DenseBlock_RNN(num_layers, num_features, growth_rate=growth_rate, **kwargs)
            )
            num_features = num_features + num_layers * growth_rate

            self.add_module("transition%d" % i, Sequential_RNN(num_features, num_features // 2, 1, **kwargs))
            num_features = num_features // 2
        self.add_module("final", SeqLinear(num_features, output_size, hidden_layer=0))


class SeperateRNN(nn.Module):
    """RNN that processes input channel groups separately before merging.

    Args:
        input_list: list of index lists, each defining a group of input channels.
        output_size: number of output features per timestep.
        num_layers: number of stacked RNN layers in the merging RNN.
        hidden_size: total hidden size (split evenly across per-group RNNs).
        linear_layers: number of hidden linear layers in the output head.
        **kwargs: additional keyword arguments forwarded to ``RNN``.
    """

    def __init__(
        self,
        input_list: list[list[int]],
        output_size: int,
        num_layers: int = 1,
        hidden_size: int = 100,
        linear_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.input_list = input_list

        rnn_width = hidden_size // len(input_list)

        self.rnns = nn.ModuleList(
            [RNN(input_size=len(n), hidden_size=rnn_width, num_layers=1, **kwargs) for n in input_list]
        )

        self.rnn = RNN(input_size=rnn_width * len(input_list), hidden_size=hidden_size, num_layers=num_layers, **kwargs)
        self.final = SeqLinear(hidden_size, output_size, hidden_size=hidden_size, hidden_layer=linear_layers)

    def forward(self, x: Tensor):
        rnn_out = [rnn(x[..., group])[0] for rnn, group in zip(self.rnns, self.input_list)]
        out = torch.cat(rnn_out, dim=-1)
        out, _ = self.rnn(out)
        out = self.final(out)
        return out
