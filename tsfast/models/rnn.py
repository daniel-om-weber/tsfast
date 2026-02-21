"""RNN-based models for time series with regularization and DenseNet variants."""

__all__ = [
    "RNN",
    "Sequential_RNN",
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

from ..data import *
from .layers import *
from ..learner.callbacks import *
from ..learner.losses import *

from fastai.basics import *

from fastai.text.models.awdlstm import RNNDropout, WeightDropout


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
        stateful: if True, persist hidden state across forward calls.
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
        self.stateful = stateful
        self.normalization = normalization
        self.bs = 1

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
        self.reset_state()

    def forward(self, inp: Tensor, h_init: list | None = None):
        bs, seq_len, _ = inp.shape
        if h_init is None and self.stateful:
            h_init = self._get_hidden(bs)

        r_input = self.input_dp(inp) if self.input_p > 0 else inp
        full_hid, new_hidden = [], []
        #         import pdb; pdb.set_trace()
        for layer_idx, (rnn, hid_dp, nrm) in enumerate(zip(self.rnns, self.hidden_dps, self.norm_layers)):
            r_output, h = rnn(r_input.contiguous(), h_init[layer_idx] if h_init is not None else None)

            if self.normalization != "":
                r_output = nrm(r_output)

            if layer_idx != self.num_layers - 1:
                r_output = hid_dp(r_output)

            full_hid.append(r_output)
            new_hidden.append(h)
            r_input = r_output

        if self.stateful:
            self.hidden = to_detach(new_hidden, cpu=False, gather=False)
            self.bs = bs
        output = r_output if not self.ret_full_hidden else torch.stack(full_hid, 0)

        return output, new_hidden

    def _get_hidden(self, bs):
        if self.hidden is None:
            return None
        if bs != self.bs:
            return None
        if self.hidden[0][0].device != one_param(self).device:
            return None
            #         import pdb; pdb.set_trace()
        return self.hidden

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

    def reset_state(self) -> None:
        """Clear the stored hidden state."""
        self.hidden = None


class Sequential_RNN(RNN):
    """RNN variant that returns only the output tensor, discarding hidden state."""

    def forward(self, inp: Tensor, h_init: list | None = None):
        return super().forward(inp, h_init)[0]


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

    @delegates(RNN, keep=True)
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

    def forward(self, x: Tensor, h_init: list | None = None):
        out, h = self.rnn(x, h_init)
        out = self.final(out)
        return out if not self.return_state else (out, h)


from ..data.loader import get_inp_out_size


@delegates(SimpleRNN, keep=True)
def RNNLearner(
    dls: DataLoaders,
    loss_func=nn.L1Loss(),
    metrics: list | None = [fun_rmse],
    n_skip: int = 0,
    num_layers: int = 1,
    hidden_size: int = 100,
    stateful: bool = False,
    opt_func=Adam,
    cbs: list | None = None,
    input_norm: type | None = StandardScaler1D,
    output_norm: type | None = None,
    **kwargs,
):
    """Create a fastai Learner with a SimpleRNN model and standard training setup.

    Args:
        dls: fastai DataLoaders providing training and validation data.
        loss_func: loss function for training.
        metrics: list of metric functions evaluated during validation.
        n_skip: number of initial timesteps to skip in loss and metric computation.
        num_layers: number of stacked RNN layers.
        hidden_size: number of hidden units per RNN layer.
        stateful: if True, enable stateful training with TBPTT reset callbacks.
        opt_func: optimizer constructor.
        cbs: additional callbacks to include in the Learner.
        input_norm: scaler class for input normalization, or None to disable.
        output_norm: scaler class for output denormalization, or None to disable.
        **kwargs: additional keyword arguments forwarded to ``SimpleRNN``.
    """
    if cbs is None:
        cbs = []

    inp, out = get_inp_out_size(dls)
    model = SimpleRNN(inp, out, num_layers, hidden_size, stateful=stateful, **kwargs)

    # Wrap model with input normalization and optional output denormalization
    if input_norm is not None:
        norm_u, _, norm_y = dls.norm_stats
        in_scaler = input_norm.from_stats(norm_u)
        out_scaler = output_norm.from_stats(norm_y) if output_norm is not None else None
        model = NormalizedModel(model, in_scaler, out_scaler)

    skip = partial(SkipNLoss, n_skip=n_skip)

    metrics = [skip(f) for f in metrics]

    if stateful:
        cbs.append(TbpttResetCB())
        # if stateful apply n_skip with a callback for the first minibatch of a tbptt sequence
        cbs.append(SkipFirstNCallback(n_skip))
    else:
        loss_func = skip(loss_func)

    lrn = Learner(dls, model, loss_func=loss_func, opt_func=opt_func, metrics=metrics, cbs=cbs, lr=3e-3)
    return lrn


@delegates(SimpleRNN, keep=True)
def AR_RNNLearner(
    dls: DataLoaders,
    alpha: float = 0,
    beta: float = 0,
    early_stop: int = 0,
    metrics: list | None = None,
    n_skip: int = 0,
    opt_func=Adam,
    input_norm: type | None = StandardScaler1D,
    **kwargs,
):
    """Create a fastai Learner with an autoregressive RNN model.

    Args:
        dls: fastai DataLoaders providing training and validation data.
        alpha: activation regularization penalty weight.
        beta: temporal activation regularization penalty weight.
        early_stop: patience for early stopping; 0 disables early stopping.
        metrics: metric functions for validation, or None for default RMSE.
        n_skip: number of initial timesteps to skip in metric computation.
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        **kwargs: additional keyword arguments forwarded to ``SimpleRNN``.
    """
    inp, out = get_inp_out_size(dls)
    ar_model = AR_Model(SimpleRNN(inp + out, out, **kwargs), ar=False)
    rnn_module = ar_model.model.rnn

    if input_norm is not None:
        norm_u, _, norm_y = dls.norm_stats
        in_scaler = input_norm.from_stats(norm_u + norm_y)
        out_scaler = input_norm.from_stats(norm_y)
        model = NormalizedModel(ar_model, in_scaler, out_scaler)
    else:
        model = ar_model

    cbs = [ARInitCB(), TimeSeriesRegularizer(alpha=alpha, beta=beta, modules=[rnn_module])]  # SaveModelCallback()
    if early_stop > 0:
        cbs += [EarlyStoppingCallback(patience=early_stop)]

    if metrics is None:
        metrics = SkipNLoss(fun_rmse, n_skip)

    lrn = Learner(dls, model, loss_func=nn.L1Loss(), opt_func=opt_func, metrics=metrics, cbs=cbs, lr=3e-3)
    return lrn


class ResidualBlock_RNN(nn.Module):
    """Two-layer RNN block with a residual skip connection.

    Args:
        input_size: number of input features per timestep.
        hidden_size: number of hidden units in each RNN layer.
        **kwargs: additional keyword arguments forwarded to ``RNN``.
    """

    @delegates(RNN, keep=True)
    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.rnn1 = RNN(input_size, hidden_size, num_layers=1, **kwargs)
        self.rnn2 = RNN(hidden_size, hidden_size, num_layers=1, **kwargs)
        self.residual = SeqLinear(input_size, hidden_size, hidden_layer=0) if hidden_size != input_size else noop

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

    @delegates(ResidualBlock_RNN, keep=True)
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

    @delegates(RNN, keep=True)
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

    @delegates(DenseLayer_RNN, keep=True)
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

    @delegates(RNN, keep=True)
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

    @delegates(RNN, keep=True)
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
