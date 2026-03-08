"""Factory functions that wire up models with training configuration."""

__all__ = [
    "RNNLearner",
    "AR_RNNLearner",
    "TCNLearner",
    "CRNNLearner",
    "AR_TCNLearner",
]

import torch
from torch import nn

from .aux_losses import ActivationRegularizer, TemporalActivationRegularizer
from .learner import Learner, TbpttLearner
from .losses import fun_rmse
from .transforms import prediction_concat
from ..models.state import GraphedStatefulModel
from ..models.cnn import CRNN, TCN
from ..models.layers import AR_Model
from ..models.rnn import SimpleRNN
from ..models.scaling import ScaledModel, Scaler, StandardScaler
from ..tsdata import get_io_size


def RNNLearner(
    dls,
    loss_func=nn.L1Loss(),
    metrics: list | None = None,
    n_skip: int = 0,
    num_layers: int = 1,
    hidden_size: int = 100,
    sub_seq_len: int | None = None,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler,
    output_norm: type | None = None,
    augmentations: list | None = None,
    transforms: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    cuda_graph: bool = False,
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
        sub_seq_len: sub-sequence length for TBPTT; enables stateful training when set.
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        output_norm: scaler class for output denormalization, or None to disable.
        augmentations: list of augmentation transforms (train only).
        transforms: list of transforms (train + valid).
        aux_losses: list of auxiliary loss functions.
        grad_clip: max gradient norm for clipping, or None to disable.
        cuda_graph: if True and sub_seq_len is set, wrap the model in GraphedStatefulModel for faster training.
        **kwargs: additional keyword arguments forwarded to ``SimpleRNN``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    if sub_seq_len:
        kwargs.setdefault("return_state", True)
    model = SimpleRNN(inp, out, num_layers, hidden_size, **kwargs)
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    if sub_seq_len and cuda_graph:
        model = GraphedStatefulModel(model)
    cls = TbpttLearner if sub_seq_len else Learner
    extra = {"sub_seq_len": sub_seq_len} if sub_seq_len else {}
    return cls(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        n_skip=n_skip,
        opt_func=opt_func,
        lr=3e-3,
        augmentations=augmentations,
        transforms=transforms,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        **extra,
    )


def AR_RNNLearner(
    dls,
    alpha: float = 0,
    beta: float = 0,
    metrics: list | None = None,
    n_skip: int = 0,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler,
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

    inp, out = get_io_size(dls)
    ar_model = AR_Model(SimpleRNN(inp + out, out, **kwargs), ar=False)
    rnn_module = ar_model.model.rnn

    model = ScaledModel.from_dls(ar_model, dls, input_norm, autoregressive=True)

    return Learner(
        model,
        dls,
        loss_func=nn.L1Loss(),
        metrics=metrics,
        n_skip=n_skip,
        opt_func=opt_func,
        lr=3e-3,
        transforms=[prediction_concat(t_offset=0)],
        aux_losses=[
            ActivationRegularizer(modules=[rnn_module], alpha=alpha),
            TemporalActivationRegularizer(modules=[rnn_module], beta=beta),
        ],
    )


def TCNLearner(
    dls,
    num_layers: int = 3,
    hidden_size: int = 100,
    loss_func: nn.Module = nn.L1Loss(),
    metrics: list | None = None,
    n_skip: int | None = None,
    opt_func: type = torch.optim.Adam,
    input_norm: type[Scaler] | None = StandardScaler,
    output_norm: type[Scaler] | None = None,
    **kwargs,
):
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
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    return Learner(model, dls, loss_func=loss_func, opt_func=opt_func, metrics=metrics, n_skip=n_skip, lr=3e-3)


def CRNNLearner(
    dls,
    loss_func: nn.Module = nn.L1Loss(),
    metrics: list | None = None,
    n_skip: int = 0,
    opt_func: type = torch.optim.Adam,
    input_norm: type[Scaler] | None = StandardScaler,
    output_norm: type[Scaler] | None = None,
    **kwargs,
):
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
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    return Learner(model, dls, loss_func=loss_func, opt_func=opt_func, metrics=metrics, n_skip=n_skip, lr=3e-3)


def AR_TCNLearner(
    dls,
    hl_depth: int = 3,
    alpha: float = 1,
    beta: float = 1,
    metrics: list | None = None,
    n_skip: int | None = None,
    opt_func: type = torch.optim.Adam,
    input_norm: type[Scaler] | None = StandardScaler,
    **kwargs,
):
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

    model = ScaledModel.from_dls(ar_model, dls, input_norm, autoregressive=True)

    return Learner(
        model,
        dls,
        loss_func=nn.L1Loss(),
        opt_func=opt_func,
        metrics=metrics,
        n_skip=n_skip,
        lr=3e-3,
        transforms=[prediction_concat(t_offset=0)],
        aux_losses=[
            ActivationRegularizer(modules=[conv_module], alpha=alpha),
            TemporalActivationRegularizer(modules=[conv_module], beta=beta),
        ],
    )
